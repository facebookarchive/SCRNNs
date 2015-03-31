--
--  Copyright (c) 2015, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
--  Author: Sumit Chopra <spchopra@fb.com>
--          Michael Mathieu <myrhev@fb.com>
--          Marc'Aurelio Ranzato <ranzato@fb.com>
--          Tomas Mikolov <tmikolov@fb.com>
--          Armand Joulin <ajoulin@fb.com>


-- Build the word based dataset for a text corpus. Words are clustered
-- based on their frequency to generate buckets with equal probability

require('math')
local ffivector = require('fb.ffivector')
local pl = require('pl.import_into')()

local Tokenizer = {}

function Tokenizer.build_dictionary(config, trainfname)
    local kMaxDictSize = 500000
    local dict = {}
    dict.symbol_to_index = {}   -- string -> id
    dict.index_to_symbol = {}   -- id -> string
    dict.index_to_freq = torch.Tensor(kMaxDictSize) -- id ->freq
    dict.index_to_cluster = nil -- id ->cluster_id
    dict.index_to_index_within_cluster = nil
    dict.cluster_to_index = {} -- reverse mapping from cluster to word id.
    dict.mapping = nil -- cluster_id to within_cluster_id mapping used by hsm

    local nr_clusters = config.nclusters
    local threshold = config.threshold

    print("[loading: " .. trainfname .. "]")
    local nr_words = 1 -- number of unique words
    local tot_nr_words = 0 -- total number of words in corpus
    -- Add by default an UNK token to be used for the rare entries
    local unk = "<UNK>"
    dict.symbol_to_index[unk] = nr_words
    dict.index_to_symbol[nr_words] = unk
    dict.index_to_freq[nr_words] = 1
    local cnt = 0
    for s in io.lines(trainfname) do
        -- remove all the tabs in the string
        s = s:gsub("\t", "")
       -- convert multiple spaces into a single space: this is needed to
       -- make the following pl.utils.split() function return only words
       -- and not white spaes
       s = s:gsub("%s+", " ")
        local words = pl.utils.split(s, ' ')
        for i, word in pairs(words) do
            if word ~= "" then -- somehow the first token is always ""
                if dict.symbol_to_index[word] == nil then
                    nr_words = nr_words + 1
                    dict.symbol_to_index[word] = nr_words
                    dict.index_to_symbol[nr_words] = word
                    dict.index_to_freq[nr_words] = 1
                else
                    local indx = dict.symbol_to_index[word]
                    dict.index_to_freq[indx] = dict.index_to_freq[indx] + 1
                end
                cnt = cnt + 1
            end
        end
        -- Add \n after every line
        if dict.symbol_to_index["</s>"] == nil then
            nr_words = nr_words + 1
            dict.symbol_to_index["</s>"] = nr_words
            dict.index_to_symbol[nr_words] = "</s>"
            dict.index_to_freq[nr_words] = 1
        else
            local indx = dict.symbol_to_index["</s>"]
            dict.index_to_freq[indx] = dict.index_to_freq[indx] + 1
        end
        cnt = cnt + 1
    end
    dict.index_to_freq:resize(nr_words)
    tot_nr_words = dict.index_to_freq:sum()
    print("[Done making the dictionary. There are " .. nr_words - 1 ..
              " unique words and a total of " .. tot_nr_words - 1 ..
              " words in the training set.]")

    -- map rare words to special token and skip corresponding indices
    -- if the specified threshold is greater than 0
    local removed = 0
    local net_nwords = 1
    if threshold > 0 then
        for i = 2, dict.index_to_freq:size(1) do
            local word = dict.index_to_symbol[i]
            if dict.index_to_freq[i] < threshold then
                dict.index_to_freq[1] =
                    dict.index_to_freq[1] + dict.index_to_freq[i]
                dict.index_to_freq[i] = 0
                dict.symbol_to_index[word] = 1
                removed = removed + 1
            else
                -- re-adjust the indices to make them continuous
                net_nwords = net_nwords + 1
                dict.index_to_freq[net_nwords] = dict.index_to_freq[i]
                dict.symbol_to_index[word] = net_nwords
                dict.index_to_symbol[net_nwords] = word
            end
        end
        print('[Removed ' .. removed .. ' rare words. ' ..
                  'Effective number of words ' .. net_nwords .. ']')
        dict.index_to_freq:resize(net_nwords)
    else
        net_nwords = nr_words
    end

    -- create the cluster index tensors
    dict.index_to_cluster = torch.LongTensor(net_nwords):fill(0)
    dict.index_to_index_within_cluster = torch.LongTensor(net_nwords):fill(0)

    -- sort the tokens by frequency
    local sorted_freqs, sorted_indx = torch.sort(dict.index_to_freq, true)
    sorted_freqs:div(math.max(1, tot_nr_words))
    if nr_clusters == 0 then
        nr_clusters = math.floor(math.sqrt(net_nwords))
    end
    local probab_mass = 1.0 / nr_clusters
    local current_mass = 0
    local cluster_id = 1
    local within_cluster_index = 0
    for w = 1, net_nwords do
        if current_mass < probab_mass then
            current_mass = current_mass + sorted_freqs[w]
            within_cluster_index = within_cluster_index + 1
        else
            cluster_id = cluster_id + 1
            current_mass = sorted_freqs[w]
            within_cluster_index = 1
        end
        dict.index_to_cluster[sorted_indx[w]] = cluster_id
        dict.index_to_index_within_cluster[sorted_indx[w]] =
            within_cluster_index
    end
    print("[Created " .. cluster_id .. " clusters.]")

    -- Count how many words per cluster there are.
    local wordsPerCluster = torch.zeros(cluster_id)
    for w = 1, net_nwords do
        local curr_cluster = dict.index_to_cluster[w]
        wordsPerCluster[curr_cluster] = wordsPerCluster[curr_cluster] + 1
    end

    -- build reverse index from cluster id back to index
    -- also load the explicit mapping to be used by hsm
    dict.mapping = torch.LongTensor(net_nwords, 2)
    for c = 1, cluster_id do
        table.insert(dict.cluster_to_index,
                     torch.LongTensor(wordsPerCluster[c]))
    end
    for w = 1, net_nwords do
        local curr_cluster = dict.index_to_cluster[w]
        local curr_word = dict.index_to_index_within_cluster[w]
        dict.cluster_to_index[curr_cluster][curr_word] = w
        dict.mapping[w][1] = curr_cluster
        dict.mapping[w][2] = curr_word
    end
    dict.separatorIndex = dict.symbol_to_index['</s>']

    -- Save dictionary.
    local dictfname = paths.concat(config.dest_path,
                                   config.name .. '.dictionary' ..
                                       '_nclust=' .. nr_clusters ..
                                       '_thresh=' .. config.threshold ..
                                       '.th7')

    torch.save(dictfname, dict)
    print('There are effectively ' .. net_nwords .. ' words in the corpus.')

    -- return the dictionary
    return dict, nr_clusters
end



-- This function tokenizes the data (converts words to word_ids)
-- and stores the result in a 1D longTensor
-- Inputs:
--          dict: dictionary
--    filenameIn: full path of the input file
--   filenameOut: full path of the output file
--        config: configuration parameters of the data
--         shuff: whether to shuffle the sentences or not (true|false)
function Tokenizer.tokenize(dict, filenameIn, filenameOut, config, eos)
   print("saving to " .. filenameOut)
   local unk = "<UNK>"
   local threshold = config.threshold
   local eos = config.eos
   -- first count how many words there are in the corpus
   local all_lines = ffivector.new_string()
   local tot_nr_words = 0
   local tot_lines = 0
   for s in io.lines(filenameIn) do
       -- store the line
       tot_lines = tot_lines + 1
       all_lines[tot_lines] = s
       -- remove all the tabs in the string
       s = s:gsub("\t", "")
       -- remove leading and following white spaces
       s = s:gsub("^%s+", ""):gsub("%s+$", "")
       -- convert multiple spaces into a single space: this is needed to
       -- make the following pl.utils.split() function return only words
       -- and not white spaes
       s = s:gsub("%s+", " ")
       -- count the words
       local words = pl.utils.split(s, ' ')
       tot_nr_words = tot_nr_words + #words -- nr. words in the line
       tot_nr_words = tot_nr_words + 1 -- newline
   end
   print('-- total lines: ' .. tot_lines)
   -- get the permutation vector
   -- perm_vec
   local perm_vec
   if shuff == true then
       print('-- shuffling the data')
       perm_vec = torch.randperm(tot_lines)
   else
       print('-- not shuffling the data')
       perm_vec = torch.range(1, tot_lines)
   end
   -- now store the lines in the tensor
   local data = torch.Tensor(tot_nr_words) -- id, cluster_id, within_cluster_id
   local id = 0
   local cnt = 1
   for ln = 1, tot_lines do
       local s = all_lines[perm_vec[ln]]
       -- remove all the tabs in the string
       s = s:gsub("\t", "")
       -- remove leading and following white spaces
       s = s:gsub("^%s+", ""):gsub("%s+$", "")
       -- convert multiple spaces into a single space: this is needed to
       -- make the following pl.utils.split() function return only words
       -- and not white spaes
       s = s:gsub("%s+", " ")
       collectgarbage()
       local words = pl.utils.split(s, ' ')
       for i, word in pairs(words) do
           if word ~= "" then
               if dict.symbol_to_index[word] == nil or
               dict.index_to_freq[dict.symbol_to_index[word]] < threshold then
                   print('WARNING: ' .. word .. ' being replaced by ' .. unk)
                   id = dict.symbol_to_index[unk]
               else
                   id = dict.symbol_to_index[word]
               end
               data[cnt] = id
               cnt = cnt + 1
           end
       end
       -- Add newline if specified
       if eos == true then
           id = dict.symbol_to_index["</s>"]
           data[cnt] = id
           cnt = cnt + 1
       end
       collectgarbage()
   end
   torch.save(filenameOut, data)
end

return Tokenizer

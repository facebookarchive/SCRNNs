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


-- This file loads a text dataset.
require 'torch'
require 'paths'
require 'math'
require 'xlua'

local WordDataset = torch.class('WordDataset')

-- config:
--   batch_size
--   root_path
--   train_file, valid_file, test_file
function WordDataset:__init(config)
    self.batch_size = config.batch_size
    self.root = config.root_path
    -- get the training, validation and test file names and load them
    self.sets = {}
    if config.train_file then
        self.train_file = paths.concat(self.root, config.train_file)
        self.sets['train'] = torch.load(self.train_file)
    end
    if config.valid_file then
        self.valid_file = paths.concat(self.root, config.valid_file)
        self.sets['valid'] = torch.load(self.valid_file)
    end
    if config.test_file then
        self.test_file = paths.concat(self.root, config.test_file)
        self.sets['test'] = torch.load(self.test_file)
    end
    collectgarbage()
end


-- returns the raw data for train|validation|test (given by set_name)
function WordDataset:get_set_from_name(set_name)
    local out = self.sets[set_name]
    if out == nil then
        if set_name == 'nil' then
            error('Set name is nil')
        else
            error('Unknown set name: ' .. set_name)
        end
    end
    return out
end


-- This function returns the data corresponding to train|valid|test sets.
-- <sname> is the name of the data type. The data is returned as a 2D tensor
-- of size: (N/batch_size)*batch_size, where N is the number of words.
function WordDataset:get_shard(sname)
    local set = self:get_set_from_name(sname)
    local shard_length = torch.floor(set:size(1)/self.batch_size)
    local cur_shard = torch.LongTensor(shard_length, self.batch_size)
    local offset = 1
    for i = 1, self.batch_size do
        cur_shard[{{},i}]:copy(set[{{offset,
                                     offset + shard_length - 1}}])
        offset = offset + shard_length
    end
    collectgarbage()
    collectgarbage()
    return cur_shard
end

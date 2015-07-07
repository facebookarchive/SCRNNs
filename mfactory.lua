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

-- this file builds and returns different models based on the
-- hyper-parameters specified in the config file
local models = {}

require("nn.LinearNB")
-- Only require LookupTableGPU/cunn if we have a cuda device.
if g_params.cuda_device then
    require("nn.LookupTableGPU")
else
    nn.LookupTableGPU = nn.LookupTable
end

-- function that takes three input arguments
--          params: table of hyperparameters for model (options.lua)
--            dict: the data dictionary (this should have information about the
--                  number of classes to be used, and also the mapping matrix
--                  to be potentially used by the HSM layer)
--       n_classes: (optional argument) specifies the number of output classes
-- The function returns a table consisting of two elements:
--      model_nets: the full assembeled model
--   intern_layers: pointers to the internal layers of the model (this is
--                  useful while initializing the internal layers)
function models.makeModelNets(params, dict, n_classes)
    -- create model layers
    local ncls = n_classes or dict.index_to_freq:size(1)
    local nhid = params.n_hidden
    local enc, dec, decloss
    local intern_layers = {}

    if string.find(params.name, 'srnn') then
        -- make the encoder
        enc = nn.Sequential()
        local net_parallel = nn.ParallelTable()
        local emb = nn.LookupTableGPU(ncls, nhid)
        local proj = nn.LinearNB(nhid, nhid)
        net_parallel:add(emb)
        net_parallel:add(proj)
        enc:add(net_parallel)
        enc:add(nn.CAddTable())
        if params.non_linearity == 'relu' then
            enc:add(nn.Threshold())
        elseif params.non_linearity == 'sigmoid' then
            enc:add(nn.Sigmoid())
        elseif params.non_linearity == 'tanh' then
            enc:add(nn.Tanh())
        else
            error("Unknown non linearity " .. params.non_linearity)
        end

        -- make the decoder
        if string.find(params.name, '_sm') then
            dec = nn.Sequential()
            dec:add(nn.LinearNB(nhid, ncls))
            dec:add(nn.LogSoftMax())
        elseif string.find(params.name, '_hsm') then
            decloss = nn.HSM(dict.mapping, nhid)
        else
            error('wrong model name')
        end

        intern_layers.emb = emb
        intern_layers.proj = proj

    elseif string.find(params.name, 'lstm') then
        -- Basic (simplified) LSTM layer as described in
        -- http://arxiv.org/pdf/1409.2329v1.pdf
        --
        -- The input to the encoder consists of a table
        -- {x_t, {h_{t-1}, c_{t-1}}}, where x_t is the current
        -- input (1-of-N vector), h_{t-1} is the previous set of
        -- hidden units and c_{t-1} is the vector of memory units.
        -- In a batch setting, x_t is a
        -- vector with B components, h_{t-1} and c_{t-1} have BxD
        -- entries, where B is the mini-batch size and D is the size
        -- of the hidden/memory state.
        -- The output of the encoder is nother table: {h_t, c_t},
        -- the current hidden state and the updated memory.
        -- The decoder takes the current hidden state h_t and computes
        -- the log prob over the classes o_t
        -- The computation is:
        -- i = logistic(W_{xi} x_t + W_{hi} H_{t-1})
        -- f = logistic(W_{xf} x_t + W_{hf} H_{t-1})
        -- o = logistic(W_{xo} x_t + W_{ho} H_{t-1})
        -- g = th(W_{xg} x_t + W_{hg} H_{t-1})
        -- c_t = f .* c_{t-1} + i .* g
        -- h_t = o .* th(c_t)
        -- Input Encoder: {x_t, {h_{t-1}, c_{t-1}}}
        -- Output Encoder: {h_t, c_t}
        -- Input Decoder: h_t
        -- Output Decoder: o_t
        local emb1 = nn.LookupTableGPU(ncls, nhid)
        local emb2 = nn.LookupTableGPU(ncls, nhid)
        local emb3 = nn.LookupTableGPU(ncls, nhid)
        local emb4 = nn.LookupTableGPU(ncls, nhid)
        local proj1 = nn.Linear(nhid, nhid)
        local proj2 = nn.Linear(nhid, nhid)
        local proj3 = nn.Linear(nhid, nhid)
        local proj4 = nn.Linear(nhid, nhid)

        -- construct the LSTM graph: encoder
        local lstm_symbol = nn.Identity()()
        local lstm_prev_state = nn.Identity()()
        -- Get the two items from the input table: previous hidden
        -- and memory state.
        local prev_hid, prev_mem = lstm_prev_state:split(2)
        local emb1n = emb1(lstm_symbol)
        local emb2n = emb2(lstm_symbol)
        local emb3n = emb3(lstm_symbol)
        local emb4n = emb4(lstm_symbol)
        local proj1n = proj1(prev_hid)
        local proj2n = proj2(prev_hid)
        local proj3n = proj3(prev_hid)
        local proj4n = proj4(prev_hid)
        local gate_i = nn.Sigmoid()(nn.CAddTable(){proj1n, emb1n})
        local gate_f = nn.Sigmoid()(nn.CAddTable(){proj2n, emb2n})
        local gate_o = nn.Sigmoid()(nn.CAddTable(){proj3n, emb3n})
        local gate_g = nn.Tanh()(nn.CAddTable(){proj4n, emb4n})

        local new_mem = nn.CAddTable()({nn.CMulTable()({gate_f, prev_mem}),
                                        nn.CMulTable()({gate_i, gate_g})})
        local new_hid = nn.CMulTable()({gate_o, nn.Tanh()(new_mem)})
        local nextState = nn.Identity(){new_hid, new_mem}
        enc = nn.gModule({lstm_symbol, lstm_prev_state}, {nextState})
        -- make the decoder
        if string.find(params.name, '_sm') then
            dec = nn.Sequential()
            dec:add(nn.Linear(nhid, ncls))
            dec:add(nn.LogSoftMax())
        elseif string.find(params.name, '_hsm') then
            decloss = nn.HSM(dict.mapping, nhid)
        else
            error('wrong model name')
        end

        intern_layers.emb1 = emb1n
        intern_layers.emb2 = emb2n
        intern_layers.emb3 = emb3n
        intern_layers.emb4 = emb4n
        intern_layers.proj1 = proj1n
        intern_layers.proj2 = proj2n
        intern_layers.proj3 = proj3n
        intern_layers.proj4 = proj4n

    elseif string.find(params.name, 'scrnn') then
        -- Structurally Constrained RNN (scrnn) with connections from the
        -- context hidden layer to the normal hidden layer. The input to the
        -- encoder consists of a table {x_t, {h_{t-1}, c_{t-1}}}, where x_t
        -- is the current input (1-of-N vector), h_{t-1} is the previous fast
        -- hidden state, and c_{t-1} is the previous context hidden state.
        -- In a batch setting
        -- x_t is a vector of B components, h_{t-1} BxD1 entires, and c_{t-1}
        -- has BxD2 entires, where B is the mini-batch size and D1 is the size
        -- of the fast hidden states and D2 is the size of the slow hidden
        -- states. The output of the encoder is another table: {h_t, c_t},
        -- the current normal hidden state and the current context hidden state.
        -- The decoder takes the current normal and context hidden states and
        -- computes the log prob over the classes o_t.
        -- The computation is:
        -- c_t = W_{xs}*x_t + W_{ss}*c_{t-1}
        -- h_t = logistic(W_{xf}*x_t + W_{ff}*h_{t-1} + W_{sf}*c_t)
        --  Input Encoder: {x_t, {h_{t-1}, c_{t-1}}}
        -- Output Encoder: {h_t, c_t}
        --  Input Decoder: {h_t, c_t}
        -- Output Decoder: o_t
        local nslow = params.n_slow
        local scale = params.semb_scale
        -- make the encoder
        local emb_fast = nn.LookupTableGPU(ncls, nhid)
        local emb_slow = nn.Sequential()
        emb_slow:add(nn.LookupTableGPU(ncls, nslow))
        emb_slow:add(nn.MulConstant(scale))
        local proj_fast = nn.LinearNB(nhid, nhid)
        local proj_slow = nn.LinearNB(nslow, nslow)
        local proj_slow2fast = nn.LinearNB(nslow, nhid)

        -- construct the scrnn encoder graph
        local input_symbol = nn.Identity()()
        local srnn_prev_state = nn.Identity()()
        -- Get the previous fast hidden and previous slow hidden
        local prev_hid_fast, prev_hid_slow = srnn_prev_state:split(2)

        local emb_fastn = emb_fast(input_symbol)
        local proj_fastn = proj_fast(prev_hid_fast)

        local emb_slown = emb_slow(input_symbol)
        local proj_slown = proj_slow(prev_hid_slow)
        local proj_slow2fastn = proj_slow2fast(prev_hid_slow)

        local new_hid_slow = nn.CAddTable()({emb_slown,
                                             proj_slown})
        local new_hid_fast = nn.Sigmoid()(nn.CAddTable(){emb_fastn,
                                                         proj_fastn,
                                                         proj_slow2fastn})
        local srnn_new_state = nn.Identity(){new_hid_fast, new_hid_slow}
        enc = nn.gModule({input_symbol, srnn_prev_state}, {srnn_new_state})

        -- make the decoder
        if string.find(params.name, '_sm') then
            dec = nn.Sequential()
            dec:add(nn.JoinTable(2))
            dec:add(nn.LinearNB(nhid + nslow, ncls))
            dec:add(nn.LogSoftMax())
        elseif string.find(params.name, '_hsm') then
            local join = nn.JoinTable(2)
            local hsm  = nn.HSM(dict.mapping, nhid + nslow)
            decloss = nn.SequentialCriterion(join:clone(), hsm:clone())
        else
            error('wrong model name')
        end

        intern_layers.emb_fast = emb_fastn
        intern_layers.emb_slow = emb_slown
        intern_layers.proj_fast = proj_fastn
        intern_layers.proj_slow = proj_slown
        intern_layers.proj_slow2fast = proj_slow2fastn
    end

    -- assemble the nets and return
    local model_nets = {
        encoder = enc,
        decoder = dec,
        decoder_with_loss = decloss
    }

    return model_nets, intern_layers
end


return models

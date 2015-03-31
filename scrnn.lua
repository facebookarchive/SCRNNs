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


-- The LSTM class which extends from RNN class. The main functions
-- are newInputTrain (to train) and test (to do only inference).

require('torch')
require('sys')
require('nn')
require('rnn')

local SCRNN = torch.class("SCRNN", "RNN")

-- config:
--   n_hidden     : number of hidden units (size of the state)
--   initial_val  : value of the initial state before any input
--   backprop_freq: number of steps between two backprops and parameter updates
--   backprop_len : number of backward steps during each backprop
--                  (should be >= backprop_freq)
--   nets         : table containing the networks:
--     encoder    : the encoder which produces a hidden state and memory
--                : state using the current input and previous hidden and
--                : memory state
--     decoder    : transformation applied to the current hidden state
--                  to produce output vector (the next symbol)
--
--                                   y1                          y2
--                                    ^                           ^
--                                  decoder                     decoder
--                                    ^                           ^
-- ... {h0, c0} -> srnn_encoder -> {h1, c1} -> srnn_encoder -> {h2, c2} ->
--                      ^                           ^
--                     x1                          x2
-- criterion     : the loss function to be used
-- ilayers       : table of pointers to internal nodes of the encoder graph.
--                 This is used so keep track of the weights and gradients
--                 associated with various internal nodes of the encoder.
--                 For instance this is required during initialization phase.
function SCRNN:__init(config, nets, criterion, ilayers)
   self.n_hidden = config.n_hidden
   self.n_slow = config.n_slow
   self.nets = {encoder = nets.encoder:clone()}
   if nets.decoder ~= nil then
       self.nets.decoder = nets.decoder:clone()
       self.criterion = criterion:clone()
   else
       assert(nets.decoder_with_loss ~= nil)
       self.nets.decoder_with_loss = nets.decoder_with_loss:clone()
   end
   self.type = torch.Tensor():type()
   self.initial_val = config.initial_val
   self.backprop_freq = config.backprop_freq
   self.batch_size = config.batch_size
   self.cuda_device = config.cuda_device
   if self.cuda_device then
        self:cuda()
   end
   self:unroll(config.backprop_len)
   self:recomputeParameters()
   self:reset()

   -- set the clipping function
   local scale_clip = function(dat, th)
       local dat_norm = dat:norm()
       if dat_norm > th then
           dat:div(dat_norm/th)
       end
   end
   local hard_clip = function(vec, th)
       local tmp = vec:float()
       local tmpp = torch.data(tmp)
       for i = 0, tmp:size(1) - 1 do
           if tmpp[i] < - th then
               tmpp[i] = - th
           else
               if tmpp[i] > th then
                   tmpp[i] = th
               end
           end
       end
       vec[{}] = tmp[{}]
   end
   if config.clip_type == 'scale' then
       self.clip_function = scale_clip
   elseif config.clip_type == 'hard' then
       self.clip_function = hard_clip
   else
       error('wrong clip type: ' .. config.clip_type)
   end

   self:set_internal_layers(ilayers)
   self.semb_scale = config.semb_scale
end


-- set the pointers to the embedding and projection modules
function SCRNN:set_internal_layers(layers)
    self.ilayers = {}
    for name, node in pairs(layers) do
        local id = node.id
        self.ilayers[name] = self.nets.encoder.fg.nodes[id].data.module
    end
end


-- call this function if you change the network architecture after creating it
-- (probably a bad idea)
function SCRNN:recomputeParameters()
    local dummy = nn.Sequential()
    for k,v in pairs(self.nets) do
        -- If we are using nn.HSM, the parameters are updated at each
        -- accGradParameters (direct_update mode). Therefore we do not
        -- add the parameters to the vector of parameters
        if torch.typename(v) ~= 'nn.SequentialCriterion' then
           dummy:add(v)
        end
    end
    -- A map from gradTensor to corresponding momentum vector
    self.w, self.dw = dummy:getParameters()
    self.mom = torch.Tensor(self.dw:size()):zero():type(self.type)
    self:unroll(#self.unrolled_nets)
end


-- ship the model to gpu
function SCRNN:cuda()
    self.type = 'torch.CudaTensor'
    if self.criterion then
        self.criterion = self.criterion:cuda()
    else
        self.nets.decoder_with_loss.module:cuda()
        self.nets.decoder_with_loss.criterion:cuda()
    end
    for _, v in pairs(self.nets) do
        v = v:cuda()
    end
    if self.unrolled_nets then
        for i = 1,#self.unrolled_nets do
            for k, v in pairs(self.unrolled_nets[i]) do
                if k ~= 'decoder_gradInput' then
                    v = v:cuda()
                else
                    for j = 1, #v do
                        v[j] = v[j]:cuda()
                    end
                end
            end
        end
    end
end


-- the user shouldnt have to manually call this function
function SCRNN:unroll(n)
    self.unrolled_nets = {}
    local params, gradParams = self.nets.encoder:parameters()
    local mem = torch.MemoryFile('w'):binary()
    mem:writeObject(self.nets.encoder)
    for i = 1, n do
        self.unrolled_nets[i] = {}
        self.unrolled_nets[i].decoder_gradInput = {}
        self.unrolled_nets[i].decoder_gradInput[1] = torch.Tensor():type(self.type)
        self.unrolled_nets[i].decoder_gradInput[2] = torch.Tensor():type(self.type)
        local reader = torch.MemoryFile(mem:storage(), 'r'):binary()
        local clone = reader:readObject()
        reader:close()
        local cloneParams, cloneGradParams = clone:parameters()
        for j = 1, #params do
            cloneParams[j]:set(params[j])
            cloneGradParams[j]:set(gradParams[j])
        end
        self.unrolled_nets[i]['encoder'] = clone
        collectgarbage()
    end
    mem:close()
end


-- returns a table of two pre-filled initial state {fast_hid, slow_hid}
function SCRNN:get_initial_state(bsize)
    if not self.initial_state then
        self.initial_state = {}
        self.initial_state[1] = torch.Tensor(bsize, self.n_hidden):type(self.type)
        self.initial_state[2] = torch.Tensor(bsize, self.n_slow):type(self.type)
        self.initial_state[1]:fill(self.initial_val)
        self.initial_state[2]:fill(self.initial_val)
    end
    return self.initial_state
end


-- Runs forward pass in the set of networks nets, with previous state prev_state
function SCRNN:elemForward(nets, input, prev_state, target)
   local bsize = input:size(1)
   prev_state = prev_state or self:get_initial_state(bsize)

   -- store the local inputs and previous state
   nets.input = input
   nets.prev_state = prev_state
   local out_encoder = nets.encoder:forward{input, prev_state}
   local out_decoder, err, n_valid = nil, nil, nil
   if self.nets.decoder ~= nil then --using the main net (not unrolled)
       assert(self.nets.decoder_with_loss == nil)
       out_decoder = self.nets.decoder:forward(out_encoder)
       if target then
           err, n_valid = self.criterion:forward(out_decoder, target)
       end
   else
       assert(self.nets.decoder_with_loss ~= nil)
       err, n_valid = self.nets.decoder_with_loss:forward(out_encoder, target)
   end
   n_valid = n_valid or input:size(1)
   return out_decoder, out_encoder, err, n_valid
end


-- Runs backward pass on the decode+criterion (or decode_with_loss) modules
function SCRNN:elemDecodeBackward(nets, target, learning_rate)
    if self.nets.decoder ~= nil then
        assert(self.nets.decoder_with_loss == nil)
        local decoder_output = self.nets.decoder.output
        local derr_do = self.criterion:backward(decoder_output, target)
        local gradInput = self.nets.decoder:backward(nets.encoder.output,
                                                     derr_do)
        nets.decoder_gradInput[1]:resizeAs(gradInput[1]):copy(gradInput[1])
        nets.decoder_gradInput[2]:resizeAs(gradInput[2]):copy(gradInput[2])
    else
        assert(self.nets.decoder_with_loss ~= nil)
        local gradInput =
            self.nets.decoder_with_loss:updateGradInput(nets.encoder.output,
                                                        target)
        nets.decoder_gradInput[1]:resizeAs(gradInput[1]):copy(gradInput[1])
        nets.decoder_gradInput[2]:resizeAs(gradInput[2]):copy(gradInput[2])
        -- This assumes the module has direct_update mode. Only HSM does:
        assert(torch.typename(self.nets.decoder_with_loss) ==
                   'nn.SequentialCriterion')
        self.nets.decoder_with_loss:zeroGradParameters()
        self.nets.decoder_with_loss:accGradParameters(nets.encoder.output, target)
        self.nets.decoder_with_loss.criterion.class_grad_bias:zero()
        self.nets.decoder_with_loss.criterion.cluster_grad_bias:zero()
        self.nets.decoder_with_loss:updateParameters(learning_rate)
    end
end


-- Main train function:
--   input : input word or minibatch
--   label : target word or minibatch
--   params:
--     learning_rate : learning rate
--     gradient_clip : if not nil, if the norm of the gradient is larger than
--                     this number, project the gradients on the sphere
--                     with this radius
-- It returns the sum of the errors and the number of terms in this sum
function SCRNN:newInputTrain(input, label, params)
   self.i_input = self.i_input + 1
   local last_nets = self.unrolled_nets[1]
   for i = 1, #self.unrolled_nets - 1 do
      self.unrolled_nets[i] = self.unrolled_nets[i + 1]
   end
   self.unrolled_nets[#self.unrolled_nets] = last_nets
   local output, next_state, err, n_valid = self:elemForward(last_nets, input,
                                                             self.state, label)
   self:elemDecodeBackward(last_nets, label, params.learning_rate)
   self.state = {}
   self.state[1] = next_state[1]
   self.state[2] = next_state[2]

   if self.i_input % self.backprop_freq == 0 then
       local gi_state_i = {}
       gi_state_i.hid = nil
       gi_state_i.slow = nil
       local unroll_bound = math.max(1, #self.unrolled_nets - self.i_input + 1)
       local j = 1
       for i = #self.unrolled_nets, unroll_bound, -1 do
           local nets = self.unrolled_nets[i]
           local prev_state_i, input_i = nets.prev_state, nets.input
           local gi_decoder_net = nets.decoder_gradInput

           -- bprop gradient from decoder for fast and slow
           if not gi_state_i.hid then
               gi_state_i.hid = gi_decoder_net[1]
               gi_state_i.slow = gi_decoder_net[2]
           elseif j <= self.backprop_freq then
               gi_state_i.hid:add(gi_decoder_net[1])
               gi_state_i.slow:add(gi_decoder_net[2])
               j = j + 1
           else
               -- do nothing, since gradients from decoder have
               -- already been accounted for
           end

           -- clip the gradients wrt hidden states
           if params.gradInput_clip then
               self:clipGradHiddens(gi_state_i.hid, params.gradInput_clip)
               self:clipGradHiddens(gi_state_i.slow, params.gradInput_clip)
           end
           -- bprop through encoder
           if i ~= 1  then
               local gi_encoder_net =
                   nets.encoder:backward({input_i, prev_state_i},
                                         {gi_state_i.hid, gi_state_i.slow})
               gi_state_i.hid = gi_encoder_net[2][1]
               gi_state_i.slow = gi_encoder_net[2][2]
           end
       end

      -- clip the gradients if specified
      if params.gradient_clip then
          self:clipGradParams(params.gradient_clip)
      end

      -- scale or zero the gradients of the slow part
      self:zeroSomeGradients()
      self:updateParams(self.w, params)
      -- zero the gradients for the next time
      self.dw:zero()
   end

   return err, n_valid
end

function SCRNN:zeroSomeGradients()
    -- slow projection gradient
    self.ilayers.proj_slow.gradWeight:zero()
end



-- Runs only forward on inputs (1d or 2d sequence of inputs) and compares
-- with labels
-- It returns the sum of the errors and the number of terms in this sum
function SCRNN:test(inputs, labels)
    local total_err = 0
    local total_n_valid = 0
    for i = 1, inputs:size(1) do
        local output, next_state, err, n_valid = self:elemForward(
           self.unrolled_nets[1], inputs[i], self.state, labels[i])
        self.state = {}
        self.state[1] = next_state[1]
        self.state[2] = next_state[2]
        if type(err) ~= 'number' then
            err = err[1]
        end
        total_err = total_err + err
        total_n_valid = total_n_valid + n_valid
    end
    return total_err, total_n_valid
end

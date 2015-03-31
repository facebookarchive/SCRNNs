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


-- This file contains a class RNN. It implements modular rnn. The main functions
-- are newInputTrain (to train) and test (to do only inference).

require 'torch'
require 'sys'
require 'nn'

local RNN = torch.class("RNN")

-- config:
--   n_hidden     : number of hidden units (size of the state)
--   initial_val  : value of the initial state before any input
--   backprop_freq: number of steps between two backprops and parameter updates
--   backprop_len : number of backward steps during each backprop
--                  (should be >= backprop_freq)
--   nets         : table containing the networks:
--     encoder    : the encoder which produces a hidden state using current
--                  input and previous hidden state
--     decoder   : transformation applied to the current hidden state
--                  to produce output vector (the next symbol)
--
--              y1               y2              y3
--              ^                ^               ^
--           decoder          decoder         decoder
--              ^                ^               ^
-- ... h0 -> encoder -> h1 -> encoder -> h2 -> encoder -> h3
--              ^                ^               ^
--              x1               x2              x3
function RNN:__init(config, nets, criterion)
   self.n_hidden = config.n_hidden
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
   self.initial_state_dim = config.initial_state_dim
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

   self:set_internal_layers()
end


function RNN:set_internal_layers()
    self.ilayers = {}
    self.ilayers.emb = self.nets.encoder.modules[1].modules[1]
    self.ilayers.proj = self.nets.encoder.modules[1].modules[2]
end


-- Reset network (training parameters and hidden state)
function RNN:reset()
   self.i_input = 0
   self.dw:zero()
   self.state = nil
end


-- ship the model to gpu
function RNN:cuda()
    self.type = 'torch.CudaTensor'
    if self.criterion then
        self.criterion = self.criterion:cuda()
    else
        self.nets.decoder_with_loss:cuda()
    end
    for _, v in pairs(self.nets) do
        v = v:cuda()
    end
    if self.unrolled_nets then
        for i = 1,#self.unrolled_nets do
            for _, v in pairs(self.unrolled_nets[i]) do
                v = v:cuda()
            end
        end
    end
end

-- returns a clone of the RNN (with the same parameter values but in
-- different storages)
function RNN:clone()
   local f = torch.MemoryFile("rw"):binary()
   f:writeObject(self)
   f:seek(1)
   local clone = f:readObject()
   f:close()
   return clone
end


-- call this function if you change the network architecture after creating it
-- (probably a bad idea)
function RNN:recomputeParameters()
    local dummy = nn.Sequential()
    for k,v in pairs(self.nets) do
        -- If we are using nn.HSM, the parameters are updated at each
        -- accGradParameters (direct_update mode). Therefore we do not
        -- add the parameters to the vector of parameter
        if torch.typename(v) ~= 'nn.HSM' then
           dummy:add(v)
        end
    end
    self.w, self.dw = dummy:getParameters()
    self.mom = torch.Tensor(self.dw:size()):zero():type(self.type)
    self:unroll(#self.unrolled_nets)
end


-- the user shouldnt have to manually call this function
function RNN:unroll(n)
    self.unrolled_nets = {}
    for i = 1, n do
        self.unrolled_nets[i] = {}
        self.unrolled_nets[i].decoder_gradInput =
            torch.Tensor():type(self.type)
        for k,v in pairs(self.nets) do
            if (k ~= 'decoder') and (k ~= 'decoder_with_loss') then
                self.unrolled_nets[i][k] = v:clone("weight", "gradWeight",
                                                   "bias", "gradBias")
            end
        end
    end
end


-- returns a tensor filled with initial state
function RNN:get_initial_state(bsize)
    local initial_state
    if self.initial_state_dim ~= nil then
       initial_state =
         torch.Tensor(torch.LongStorage(self.initial_state_dim)):type(self.type)
    else
       initial_state = torch.Tensor(bsize, self.n_hidden):type(self.type)
    end
    initial_state:fill(self.initial_val)
   return initial_state
end


-- Runs forward pass in the set of networks nets, with previous state prev_state
function RNN:elemForward(nets, input, prev_state, target)
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
function RNN:elemDecodeBackward(nets, target, learning_rate)
    if self.nets.decoder ~= nil then
        assert(self.nets.decoder_with_loss == nil)
        local decoder_output = self.nets.decoder.output
        local derr_do = self.criterion:backward(decoder_output, target)
        local gradInput = self.nets.decoder:backward(nets.encoder.output,
                                                     derr_do)
        nets.decoder_gradInput:resizeAs(gradInput):copy(gradInput)
    else
        assert(self.nets.decoder_with_loss ~= nil)
        local gradInput =
            self.nets.decoder_with_loss:updateGradInput(nets.encoder.output,
                                                        target)
        nets.decoder_gradInput:resizeAs(gradInput):copy(gradInput)
        -- This assumes the module has direct_update mode. Only HSM does:
        assert(torch.typename(self.nets.decoder_with_loss) == 'nn.HSM')
        -- self.nets.decoder_with_loss:zeroGradParameters()
        self.nets.decoder_with_loss:accGradParameters(
            nets.encoder.output, target, -learning_rate, true)
        self.nets.decoder_with_loss.class_grad_bias:zero()
        self.nets.decoder_with_loss.cluster_grad_bias:zero()
    end
end


-- function to update the parameters
function RNN:updateParams(w, params)
    if params.momentum then
        self.mom:mul(params.momentum)
        self.mom:add(self.dw)
        w:add(- params.learning_rate, self.mom)
    else
        w:add(- params.learning_rate, self.dw)
    end
end


-- function to clip the gradients of the parameters
function RNN:clipGradParams(gclip)
    for k,v in pairs(self.nets) do
        local lw, ldw = v:parameters()
        if ldw then
            for i = 1, #ldw do
                if ldw[i] then
                    self.clip_function(ldw[i], gclip)
                end
            end
        end
    end
end



-- function to clip the gradients of the hidden states
function RNN:clipGradHiddens(vec, gclip)
    self.clip_function(vec, gclip)
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
function RNN:newInputTrain(input, label, params)
   self.i_input = self.i_input + 1
   local last_nets = self.unrolled_nets[1]
   for i = 1, #self.unrolled_nets-1 do
      self.unrolled_nets[i] = self.unrolled_nets[i+1]
   end
   self.unrolled_nets[#self.unrolled_nets] = last_nets
   local _output, next_state, err, n_valid = self:elemForward(last_nets, input,
                                                             self.state, label)
   self:elemDecodeBackward(last_nets, label, params.learning_rate)
   self.state = next_state

   if self.i_input % self.backprop_freq == 0 then
      local inc_gi_state_i = nil
      local unroll_bound = math.max(1, #self.unrolled_nets - self.i_input + 1)
      local j = 1
      for i = #self.unrolled_nets, unroll_bound, -1 do
         local nets = self.unrolled_nets[i]
         local prev_state_i, input_i = nets.prev_state, nets.input
         local gi_decoder_net = nets.decoder_gradInput

         -- get gradients from decoder
         if not inc_gi_state_i then
             inc_gi_state_i = gi_decoder_net
         elseif j <= self.backprop_freq then
             inc_gi_state_i:add(gi_decoder_net)
             j = j + 1
         else
             -- do nothing, since gradients from decoder have already
             -- been accounted for
         end

         -- clip the gradients wrt hidden states
         if params.gradInput_clip then
             self:clipGradHiddens(inc_gi_state_i, params.gradInput_clip)
         end
         -- bprop through encoder
         if i ~= 1  then
             local gi_encoder_net =
                 nets.encoder:backward({input_i, prev_state_i}, inc_gi_state_i)
             inc_gi_state_i = gi_encoder_net[2]
         end
      end

      -- clip the gradients if specified
      if params.gradient_clip then
          self:clipGradParams(params.gradient_clip)
      end

      -- update the parameters
      self:updateParams(self.w, params)
      -- zero the gradients for the next time
      self.dw:zero()
   end

   return err, n_valid
end


-- Runs only forward on inputs (1d or 2d sequence of inputs) and compares
-- with labels
-- It returns the sum of the errors and the number of terms in this sum
function RNN:test(inputs, labels)
    local total_err = 0
    local total_n_valid = 0
    for i = 1,inputs:size(1) do
        local _output, next_state, err, n_valid = self:elemForward(
           self.unrolled_nets[1], inputs[i], self.state, labels[i])
        self.state = next_state
        if type(err) ~= 'number' then
            err = err[1]
        end
        total_err = total_err + err
        total_n_valid = total_n_valid + n_valid
    end
    return total_err, total_n_valid
end

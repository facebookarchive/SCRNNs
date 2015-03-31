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

-- This file specifies various utilities being used through the code,
-- such as initialization functions etc.
require 'math'

local utils = {}

function utils.sparseInit(w, nz, lim)
   -- Assume that the number of inputs equals the number of rows R
   -- and the number of outputs equals the number of cols C.
   -- w is a matrix of size RxC.
   -- This function connects (with non-zero weights) each output to  only nz
   -- inputs taken at random. This follows the Martens & Sutskever recipe.
   -- Each non-zero value is drawn at uniform from the interval (-lim, lim).
   local ni = w:size(1)
   local no = w:size(2)
   nz = nz and nz or 15 -- default value set to 15 like in S&M paper.
   lim = lim and lim or 1 -- default value set to 1
   w:fill(0.0)
   for o = 1, no do
      local perm = torch.randperm(ni)
      for i = 1, nz do
          w[perm[i]][o] = math.random() * lim - lim
      end
   end
end


function utils.orthInit(w, lim, scl)
    -- Assume that the weight matrix w is square with equal number of rows R
    -- (inputs) and the cols C (outputs).
    -- w is a matrix of size RxC (R=C).
    -- This function initializes w close to identity, with each of the
    -- non-diagonal elements randomly sampled from a normal distribution
    -- with 0 mean and standard deviation lim
    -- scl is the values on the diagonal. default is 0.9
    lim = lim and lim or 0 -- default value set to 0.
    scl = scl and scl or 0.9
    w:zero()
    local ni = w:size(1)
    if lim == 0 then
        w:add(torch.eye(ni):typeAs(w) * scl)
    elseif lim > 0 then
        w:normal(0, lim)
        fetch_diag(w):fill(scl)
    end
end


function utils.initRNN(params, model, ilayers)
    -- encoder
    model.w:normal():mul(0.05)
    if params.w_init == 'srand' then
        utils.sparseInit(ilayers.proj.weight, params.sparse_init_num_non_zero)
    elseif params.w_init == 'frand' then
        -- already initialized
    elseif params.w_init == 'eye' then
        utils.orthInit(ilayers.proj.weight, 0)
    end
    -- decoder
    if model.nets.decoder ~= nil then
        model.nets.decoder.modules[1].weight:zero()
    elseif g_model.nets.decoder_with_loss then
        model.nets.decoder_with_loss:reset(0., 0.)
    end
end


function utils.initLSTM(params, model, ilayers)
    -- encoder
    model.w:normal():mul(0.05)
    if params.w_init == 'srand' then
        for i = 1, 4 do
            local name = 'proj' .. i
            utils.sparseInit(ilayers[name].weight,
                             params.sparse_init_num_non_zero)
        end
    elseif params.w_init == 'frand' then
        -- already initialized
    elseif params.w_init == 'eye' then
        for i = 1, 4 do
            local name = 'proj' .. i
            utils.orthInit(ilayers[name].weight, 0)
        end
    end
    for i = 1, 4 do
        local name = 'proj' .. i
        ilayers[name].bias:zero()
    end
    -- decoder
    if model.nets.decoder ~= nil then
        model.nets.decoder.modules[1].bias:zero()
        model.nets.decoder.modules[1].weight:zero()
    elseif model.nets.decoder_with_loss then
        model.nets.decoder_with_loss:reset(0., 0.)
    end
end


function utils.initSCRNN(params, model, ilayers)
    -- encoder
    model.w:normal():mul(0.05)
    if params.w_init == 'srand' then
        utils.sparseInit(ilayers.proj_fast.weight,
                         params.sparse_init_num_non_zero)
    elseif params.w_init == 'frand' then
        -- already initialized
    elseif params.w_init == 'eye' then
        utils.orthInit(ilayers.proj_fast.weight, 0)
    end
    ---- initialize contextual part
    utils.orthInit(ilayers.proj_slow.weight, 0, 1 - params.semb_scale)

    -- decoder
    if model.nets.decoder ~= nil then
        model.nets.decoder.modules[2].weight:zero()
    elseif model.nets.decoder_with_loss then
        model.nets.decoder_with_loss.criterion:reset(0., 0.)
    end
end


return utils

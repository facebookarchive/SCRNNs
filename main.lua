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

-- This file trains and tests the RNN with single worker.
require('nn')
require('nngraph')
require('fbcunn')
require('train')
require('options')
local dtls = require('datatools')
local mdls = require('mfactory')
local utls = require('util')

-- Parse arguments
local cmd = RNNOption()
cmd:option('-overrideparams',
           'model.override_loaded_parameters',
           false, 'If loading a model, overrides its parameters')
g_params = cmd:parse(arg)
g_params.trainer.save_dir = g_params.trainer.save_dir

-- cuda?
if g_params.cuda_device then
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(g_params.cuda_device)
end

if string.find(g_params.model.name, 'srnn') then
    require('rnn')
elseif string.find(g_params.model.name, 'lstm') then
    require('lstm')
elseif string.find(g_params.model.name, 'scrnn') then
    require('scrnn')
    g_params.model.semb_scale = 0.05
else
    error('**** wrong model ****')
end

-- build the torch dataset
dtls.generate_data(g_params.dataset)

-- Load dataset and dictionary
print('[[ Loading dataset and dictionary ]]')
g_dataset, g_dictionary = dtls.load_dataset(g_params.dataset)

-- create model layers
local nets, ilayers = mdls.makeModelNets(g_params.model, g_dictionary)

-- load the loss function
local criterion
if string.find(g_params.model.name, '_sm') then
    criterion = nn.ClassNLLCriterion()
    criterion.sizeAverage = false
end

-- create model and initialize
torch.manualSeed(1)
if string.find(g_params.model.name, 'srnn') then
    g_model = RNN(g_params.model, nets, criterion)
    g_ilayers = g_model.ilayers
    utls.initRNN(g_params.model, g_model, g_ilayers)

elseif string.find(g_params.model.name, 'lstm') then
    g_model = LSTM(g_params.model, nets, criterion, ilayers)
    g_ilayers = g_model.ilayers
    utls.initLSTM(g_params.model, g_model, g_ilayers)

elseif string.find(g_params.model.name, 'scrnn') then
    g_model = SCRNN(g_params.model, nets, criterion, ilayers)
    g_ilayers = g_model.ilayers
    utls.initSCRNN(g_params.model, g_model, g_ilayers)

else
    error('*** wrong model ***')
end

-- Create trainer
g_trainer = RNNTrainer(g_params.trainer, g_model, g_dataset)
if g_params.cuda_device then
    g_trainer:cuda()
end

-- Print parameters
cmd:print_params(g_params)

-- train!
g_trainer:run(g_params.trainer.n_epochs)

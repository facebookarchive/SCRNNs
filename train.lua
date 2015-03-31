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


-- This file contains training routines when *not* using hogwild.
require 'math'
require 'sys'
require 'os'
require 'torch'
require 'xlua'

local RNNTrainer = torch.class('RNNTrainer')


-- config:
--   initial_learning_rate
--   learning_rate_shrink : divides the learning rate when the
--                         validation error increases
--   gradient_clip
--   gradInput_clip
--   momentum
--   use_valid
--   use_test
--   progress_bar
--   unk_index
--   save_dir
function RNNTrainer:__init(config, model, dataset)
   self.training_params = {learning_rate = config.initial_learning_rate,
                           gradient_clip = config.gradient_clip,
                           gradInput_clip = config.gradInput_clip,
                           momentum = config.momentum}

   self.learning_rate_shrink = config.learning_rate_shrink
   self.shrink_multiplier = config.shrink_factor
   self.trbatches = config.trbatches
   self.unk_index = config.unk_index
   self.progress_bar = not config.no_progress
   self.model = model
   self.dataset = dataset
   self.save_dir = config.save_dir
   self.use_valid = config.use_valid_set
   self.use_test = config.use_test_set
   self.type = torch.Tensor():type()
   self.anneal_type = config.shrink_type
   self.annealed = false
end

function RNNTrainer:cuda()
    self.type = 'torch.CudaTensor'
end

-- main train function
function RNNTrainer:run_epoch_train()
    local total_err = 0
    local n_total = 0
    local n_words = 0
    self.model:reset()
    local shard = self.dataset:get_shard('train')
    if self.type == 'torch.CudaTensor' then
        shard = shard:type(self.type)
    end
    local inputs = shard[{{1, shard:size(1)-1}}]
    local labels = shard[{{2, shard:size(1)}}]
    local batch_size = shard:size(2)
    n_words = n_words + inputs:size(1)*inputs:size(2)
    local size = self.trbatches == -1 and inputs:size(1) or self.trbatches
    for i = 1, size do
        if i % 1000 == 0 then
            if self.progress_bar then
                xlua.progress(i, size)
            end
            if sys.isNaN(self.model.w:sum()) then
                print('Not a Number detected')
                os.exit(0)
            end
        end
        local err, n = self.model:newInputTrain(inputs[i], labels[i],
                                                self.training_params)
        if type(err) ~= 'number' then
            err = err[1]
        end
        total_err = total_err + err
        n_total = n_total + n
    end
    collectgarbage()
    return total_err/n_total / math.log(2), n_words
end

-- validation function : runs the model on the validation set (or any set)
-- and returns the average entropy (base 2)
function RNNTrainer:run_epoch_val(set_name)
    set_name = set_name or 'valid'
    -- local n_shards = self.dataset:get_n_shards(set_name)
    local total_err, total_n = 0, 0
    self.model:reset()
    local shard = self.dataset:get_shard(set_name)
    if self.type == 'torch.CudaTensor' then
            shard = shard:type(self.type)
    end
    local inputs = shard[{{1, shard:size(1)-1}}]
    local labels = shard[{{2, shard:size(1)}}]
    local err, n = self.model:test(inputs, labels)
    total_err = total_err + err
    total_n = total_n + n
    collectgarbage()
    return total_err / total_n / math.log(2)
end

-- see run_epoch_val (same on test set)
function RNNTrainer:run_epoch_test()
    return self:run_epoch_val('test')
end

-- runs train, validation and test on n_epoches epoches
function RNNTrainer:run(n_epoches)
   local last_val_err = 1e30
   local last_model = nil
   local unk_index = self.unk_index
   local train_err = torch.zeros(n_epoches)
   local val_err = torch.zeros(n_epoches)
   local test_err = torch.zeros(n_epoches)
   local time = torch.zeros(n_epoches)

   -- save the untrained model
   if self.save_dir ~= nil then
       if paths.dirp(self.save_dir) == false then
           os.execute('mkdir -p ' .. self.save_dir)
       end
       print('*** saving the model ***')
       torch.save(paths.concat(self.save_dir, 'model_0'), self.model)
   end


   for i = 1, n_epoches do
      local timer = torch.tic()
      local n_words
      if (self.unk_index ~= nil) and
      (self.model.nets.decode_with_loss ~= nil) then
          -- UNK with HSM
          -- disable unk_index for training
          self.model.nets.decode_with_loss.unk_index = 0
          train_err[i], n_words = self:run_epoch_train()
          -- enable unk_index for testing
          self.model.nets.decode_with_loss.unk_index = unk_index
      else
          train_err[i], n_words = self:run_epoch_train()
      end
      time[i] = torch.toc(timer)
      io.write(string.format('\n\nEpoch: %d. Training time: %.2fs. ' ..
                                 'Words/s: %.2f',
                             i, time[i], n_words/time[i]))
      io.write(string.format('\nTraining: Entropy (base 2) : %.5f || ' ..
                                 'Perplexity : %0.5f',
                             train_err[i], math.pow(2, train_err[i])))
      io.flush()

      -- save the trained model
      if self.save_dir ~= nil then
          if paths.dirp(self.save_dir) == false then
              os.execute('mkdir -p ' .. self.save_dir)
          end
          torch.save(paths.concat(self.save_dir, 'model_' .. i), self.model)
      end

      -- evaluate model on the validation set
      if (self.use_valid == 1) or (self.use_valid == true) then
          val_err[i] = self:run_epoch_val('valid')
          -- io.write(string.format('Total time : %.2fs',torch.toc(timer)))
          io.write(string.format('\nValidation: Entropy (base 2) : %.5f || ' ..
                                     'Perplexity : %0.5f',
                                 val_err[i], math.pow(2, val_err[i])))
          io.flush()
      end

      -- evaluate model on the test set
      if (self.use_test == 1) or (self.use_test == true) then
          test_err [i]= self:run_epoch_test('test')
          io.write(string.format('\nTesting: Entropy (base 2) : %.5f || '..
                                  'Perplexity : %0.5f',
                              test_err[i], math.pow(2, test_err[i])))
          io.flush()
      end


      -- decrease learning rate if needed
      if self.annealed == false then
          if (self.use_valid == 1) and (self.learning_rate_shrink ~= nil) and
          (val_err[i] > last_val_err * self.shrink_multiplier) then
              self.training_params.learning_rate =
                  self.training_params.learning_rate / self.learning_rate_shrink
              self.model = last_model or self.model
              io.write('\nDecreasing the learning rate to '
                           .. self.training_params.learning_rate)
              if self.anneal_type == 'fast' then
                  self.annealed = true
              end
          else
              last_val_err = val_err[i]
              last_model = self.model:clone()
          end
      else -- anneal the learning rate after every subsequent epoch
          self.training_params.learning_rate =
              self.training_params.learning_rate / self.learning_rate_shrink
          io.write('\nDecreasing the learning rate to '
                       .. self.training_params.learning_rate)
      end

      print('')

      -- save the logs of accuracy and time
      if self.save_dir ~= nil then
         torch.save(paths.concat(self.save_dir, 'model.log'),
                    {train_err=train_err, test_err=test_err,
                     valid_err=val_err, time=time, epoch=i})
      end

   end
end


function RNNTrainer:evaluate()
    -- evaluate model on the validation set
    local val_err = self:run_epoch_val('valid')
    -- io.write(string.format('Total time : %.2fs',torch.toc(timer)))
    io.write(string.format('\nValidation: Entropy (base 2) : %.5f || ' ..
                               'Perplexity : %0.5f',
                           val_err, math.pow(2, val_err)))
    io.flush()

    -- evaluate model on the test set
    local test_err = self:run_epoch_test('test')
    io.write(string.format('\nTesting: Entropy (base 2) : %.5f || '..
                               'Perplexity : %0.5f',
                           test_err, math.pow(2, test_err)))
    io.flush()
end

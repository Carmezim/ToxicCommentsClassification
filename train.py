import numpy as np

from tqdm import tqdm
from base_train import BaseTrain
from utils import utils as u


class Train(BaseTrain):
    def __init__(self, sess, model, data, c, logger):
        super(Train, self).__init__(sess, model, data, c, logger)

    def train_epoch(self):
        cur_i = self.model.global_step_tensor.eval(self.sess).item()
        print("Training epoch %d" % self.model.cur_epoch_tensor.eval(
            self.sess).item())
        loop = tqdm(range(self.c.n_epochs))
        losses = []
        accs = []
        val_losses = []
        val_accs = []
        for i in loop:
            self.model.save(self.sess)
            loss, acc, val_loss, val_acc = self.train_step()
            losses.append(loss)
            accs.append(acc)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            print("Iter {} ".format(i) + "training loss: ", loss)
            print("Iter {} ".format(i) + "training accuracy: ", acc)
            print("Iter {} ".format(i) + "validation loss: ", val_loss)
            print("Iter {} ".format(i) + "validation accuracy: ", val_acc)
        total_loss = np.mean(losses)
        total_acc = np.mean(accs)
        total_val_loss = np.mean(val_losses)
        total_val_acc = np.mean(val_accs)
        print("Epoch {}".format(cur_i) + "total training loss: ", total_loss)
        print("Epoch {}".format(cur_i) + "total training accuracy: ", total_acc)
        print("Epoch {}".format(cur_i) +
              "total validation loss: ", total_val_loss)
        print("Epoch {}".format(cur_i) +
              "total validation accuracy: ", total_val_acc)

        summaries = dict()
        summaries['loss'] = total_loss
        summaries['acc'] = total_acc
        summaries['validation_loss'] = total_val_loss
        summaries['validation_acc'] = total_val_acc
        print("reached summaries")
        self.logger.summarize(cur_i, summaries_dict=summaries)
        self.model.save(self.sess)

    def train_step(self):
        # input_len = np.empty(self.c.b)
        # input_len.fill(self.c.max_len)
        print("minibatch training...")
        pbar = tqdm(total=int(self.data[0].shape[0] // self.c.b))
        step = 0
        for batch_x, batch_y in u.minibatches(self.data[0], self.data[1],
                                              self.c.b):

            feed_dict = {self.model.x: batch_x, self.model.y: batch_y,
                         self.model.is_training: True}

            _, loss, acc = self.sess.run([self.model.optimizer, self.model.cost,
                                          self.model.accuracy],
                                         feed_dict=feed_dict)

            val_x = self.data[3][:self.c.b]
            val_y = self.data[4][:self.c.b]
            val_feed_dict = {self.model.x: val_x, self.model.y: val_y,
                             self.model.is_training: True}

            # each 50th step compute validation loss and accuracy
            if step % 50 == 0:
                val_acc, val_loss = self.sess.run([self.model.accuracy,
                                                   self.model.cost],
                                                  feed_dict=val_feed_dict)

                print("loss shape", loss.shape)
                print("Iter {}".format(step * self.c.b) +
                      "\nTraining Loss: ", loss)
                print("Training Accuracy: ", acc)
                print("Validation Loss:", val_loss)
                print("Validation Accuracy: ", val_acc)
            step += 1
            pbar.update(1)

        return loss, acc, val_loss, val_acc

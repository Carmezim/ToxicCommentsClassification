import tensorflow as tf

from utils import utils as u
from utils.logger import Logger
from models.BiLSTM import BiLSTM
from train import Train


def main():
    sess = tf.Session()
    data = u.return_data()
    model = BiLSTM(u.Config, data[2])
    logger = Logger(sess, u.Config)
    trainer = Train(sess, model, data, u.Config, logger)
    trainer.train()


if __name__ == "__main__":
    main()

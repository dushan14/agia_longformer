import logging

import arg_parser
import longformer_test
import longformer_trainer

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

if __name__ == '__main__':
    args = arg_parser.parse_args()
    logging.info('Args : %s', args)

    if args.task == 'train':
        logging.info("Training Started")
        longformer_trainer.process(args)
        logging.info("Training Completed")

    if args.task == 'test':
        logging.info("Testing Started")
        longformer_test.process(args)
        logging.info("Testing Completed")

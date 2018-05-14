from dataloader import Dataloader
from advanced_tutorial import prepare_sequence, BiLSTM_CRF
import torch
import torch.optim as optim
import argparse
from tokenacc import TokenAcc
import logging
import os


parser = argparse.ArgumentParser()
parser.add_argument('--out', type=str, default='exp/lr1e-3')
parser.add_argument('--log-interval', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--cuda', default=True, action='store_false')
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)
logging.basicConfig(format='%(asctime)s: %(message)s', datefmt='%H:%M:%S', filename=os.path.join(args.out, 'train.log'), level=logging.INFO)
#logger_decode = logging.getLogger('decodelog', f)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
EMBEDDING_DIM = 128
HIDDEN_DIM = 32
tag_to_ix = {"O": 0, "B-treatment": 1, "B-problem": 2, "B-test": 3, "I-treatment": 4, "I-problem": 5, "I-test": 6, START_TAG: 7, STOP_TAG: 8}
ix_to_tag = {0 : "O", 1 : "B-treatment", 2: "B-problem",  3: "B-test", 4 : "I-treatment", 5: "I-problem", 6 : "I-test", 7: START_TAG, 8: STOP_TAG}
training_data = Dataloader('../data/ner/train.eval').data
dev_data = Dataloader('../data/ner/dev.eval').data

word_to_ix = {}
for sentence, tags in training_data+dev_data:
    for word in sentence:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)


lr = args.lr
model = BiLSTM_CRF(len(word_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=1e-4)
if args.cuda: model.cuda()


# train
## Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

def adjust_learning_rate(optimizer, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    # lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

prev_loss = 1e9
tokenAcc = TokenAcc()
# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(
        300):  # again, normally you would NOT do 300 epochs, it is toy data
    model.train()
    for i, (sentence, tags) in enumerate(training_data):
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        if args.cuda: sentence_in.cuda();targets.cuda()

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)
        tokenAcc.update(loss.data)
        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        loss.backward()
        optimizer.step()
        if i % args.log_interval == 0 and i > 0:
            logging.info('[Epoch %d Sample %d] loss %.2f' % (epoch, i, tokenAcc.get()))

    trainloss = tokenAcc.getAll()
    tokenAcc.reset()
    model.eval()
    for sentence, tags in dev_data:
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[t] for t in tags], dtype=torch.long)
        precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)

        #print(sentence)
        #print(ix_to_tag[x] for x in model(precheck_sent))

        if args.cuda: sentence_in.cuda(); targets.cuda()

        # Step 3. Run our forward pass.
        cvloss = model.neg_log_likelihood(sentence_in, targets)
        tokenAcc.update(cvloss)
    logging.info('[Epoch %d trloss: %.2f cvloss: %.2f]' % (epoch, trainloss, tokenAcc.getAll()))
    if cvloss < prev_loss:
        prev_loss = cvloss
        best_model = '{}/params_epoch{:02d}_tr{:.2f}_cv{:.2f}'.format(args.out, epoch, trainloss, tokenAcc.getAll())
        torch.save(model.state_dict(), best_model)
    else:
        torch.save(model.state_dict(), '{}/params_epoch{:02d}_tr{:.2f}_cv{:.2f}_rejected'.format(args.out, epoch, trainloss, tokenAcc.getAll()))
        model.load_state_dict(torch.load(best_model))
        if args.cuda: model.cuda()

        lr /= 2
        adjust_learning_rate(optimizer, lr)

# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    print(model(precheck_sent))
# We got it!


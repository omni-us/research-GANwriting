import glob
import os
import argparse

parser = argparse.ArgumentParser('delete models')
parser.add_argument('epoch', type=int, help='delete all the models < this epoch')
args = parser.parse_args()

models = glob.glob('save_weights/*.model')
for i in models:
    if int(i.split('/')[1].split('.')[0].split('-')[1]) > args.epoch:
        command = 'rm ' + i
        print(command)
        os.system(command)

preds = glob.glob('pred_logs/*predict*')
for i in preds:
    if int(i.split('/')[1].split('.')[1]) > args.epoch:
        command = 'rm ' + i
        print(command)
        os.system(command)

imgs = glob.glob('imgs/*.jpg')
for i in imgs:
    if int(i.split('/')[1].split('.')[0].split('_')[-1]) > args.epoch:
        command = 'rm ' + i
        print(command)
        os.system(command)

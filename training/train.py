"""Repository-root training entrypoint; delegates to the preserved public trainer."""
import argparse
from pathlib import Path
import sys

REPOSITORY=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPOSITORY/'backend'))
from train_energy_public import train


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv',type=Path,default=REPOSITORY/'data/ChargePoint Data CY20Q4.csv')
    parser.add_argument('--output',type=Path,required=True,help='New empty directory named energy_public_v1; existing artifacts are never overwritten.')
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--patience',type=int,default=5)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    train(args.csv,args.output,args.epochs,args.patience,args.seed)

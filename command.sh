
python main.py --dataset NCI1 --gpc 1 --nhid 64
python main.py --dataset NCI1 --gpc 10 --nhid 64
python main.py --dataset NCI1 --gpc 50 --nhid 64

python main.py --dataset ogbg-molhiv --gpc 1 --initialize Herding --outer_bs 128 --outer_lr 1e-3 --nhid 128 --train_model gin --test_model gin
python main.py --dataset ogbg-molhiv --gpc 10 --initialize Herding --outer_bs 128 --outer_lr 1e-3 --nhid 128 --train_model gin --test_model gin
python main.py --dataset ogbg-molhiv --gpc 50 --initialize Herding --outer_bs 128 --outer_lr 1e-3 --nhid 128 --train_model gin --test_model gin

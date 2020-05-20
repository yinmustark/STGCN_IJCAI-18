for rank_s in 10 20 30
do
    save_dir="./output$rank_s"
    mkdir -p $save_dir
    python main.py --rank $rank_s 3 --save_path $save_dir
done
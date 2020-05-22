for rank_s in 3 15 25
do
    save_dir="./output_new$rank_s"
    mkdir -p $save_dir
    python main.py --rank $rank_s 3 --save_path $save_dir
done

ROOT=/ssd/hust-tangxi/workspace/MOR

source ${ROOT}/.venv/bin/activate
which python

export CUDA_VISIBLE_DEVICES="1"
export NCCL_IB_DISABLE=1

dataset_name=Amazon-Reviews-2023
# * Industrial_and_Scientific Office_Products All_Beauty
domain=All_Beauty
# * Qwen3.5-0.8B Qwen2.5-Coder-7B 
model_name=Qwen3.5-0.8B 
# * Qwen3-VL-Embedding-2B Qwen3-Embedding-0.6B
embedding_model_name=Qwen3-Embedding-0.6B

black ${ROOT}/src

run_prepare() {
    cd ${ROOT}/src/data
    python amazon_data_process.py \
        --dataset ${domain} \
        --metadata_file ${ROOT}/data/${dataset_name}/raw/meta_categories/meta_${domain}.jsonl \
        --reviews_file ${ROOT}/data/${dataset_name}/raw/review_categories/${domain}.jsonl \
        --user_k 5 \
        --st_year 2018 \
        --st_month 10 \
        --ed_year 2023 \
        --ed_month 9 \
        --output_path ${ROOT}/data/${dataset_name}/splited/${domain}
    wait

    # * 生成嵌入
    export VLLM_WORKER_MULTIPROC_METHOD="spawn"
    export VLLM_ATTENTION_BACKEND="TORCH_SDPA"
    cd ${ROOT}/src/residual_quantization/item_embedding
    python amazon_textimg2emb.py \
        --dataset ${domain} \
        --plm_name ${embedding_model_name} \
        --root ${ROOT}/data/${dataset_name}/splited/${domain} \
        --plm_checkpoint ${ROOT}/pretrained-models/${embedding_model_name} \
        --batch_size 64
    wait
}

run_rqvae() {
    output_dir=/ssd/hust-tangxi/workspace/MOR/saved_models/RQ-VAE/${dataset_name}/${domain}
    # rm -rf ${output_dir}
    mkdir -p ${output_dir}
    
    cd ${ROOT}/src/residual_quantization/CF/LightGCN-PyTorch/code
    python main.py \
        --decay=1e-4 \
        --lr=0.001 \
        --layer=3 \
        --seed=2026 \
        --dataset="amazon-book" \
        --topks="[20]" \
        --recdim=64 \
        --path="$output_dir/LightGCN" \
        --bpr_batch=8192 \
        --epochs=100
    wait

    exit

    # * 优化 RQ-VAE: 残差量化自编码器
    cd ${ROOT}/src/residual_quantization
    python train_rqvae.py \
      --data_path ${ROOT}/data/${dataset_name}/splited/${domain}/${domain}.emb-${embedding_model_name}-td.npy \
      --output_dir ${output_dir} \
      --lr 1e-3 \
      --epochs 10000 \
      --batch_size 20480 \
      --eval_step 50
    wait

    exit
    
    # * 对 Title 生成 SID
    python generate_indices.py \
      --ckpt_path ${output_dir}/best_collision_model.pth \
      --output_path ${ROOT}/data/Amazon23/${domain}/${domain}.index.json
    wait
    
    # * 转化为训练所需的格式
    cd ${ROOT}
    black convert_dataset.py
    python convert_dataset.py \
        --dataset_name ${domain} \
        --data_dir ${ROOT}/data/Amazon23/${domain} \
        --output_dir ${ROOT}/datasets/Amazon23/${domain}
    wait
}

run_sft() {
    base_model_path=/ssd/hust-tangxi/workspace/pretrained-models/${model_name}
    dataset_dir=${ROOT}/datasets/Amazon23/${domain}
    output_dir=${ROOT}/saved_models/${model_name}/Amazon23/${domain}-sft
    
    n_gpus_per_node=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')
    
    # * 用监督微调对齐 SID 和语义
    cd ${ROOT}/src
    black sft.py
    torchrun --nproc_per_node ${n_gpus_per_node} \
        sft.py \
        --base_model ${base_model_path} \
        --batch_size 256 \
        --micro_batch_size 16 \
        --train_file ${dataset_dir}/train/${domain}.csv \
        --eval_file ${dataset_dir}/valid/${domain}.csv \
        --output_dir ${output_dir} \
        --category ${domain} \
        --train_from_scratch False \
        --seed 42 \
        --sid_index_path ${ROOT}/data/Amazon23/${domain}/${domain}.index.json \
        --item_meta_path ${ROOT}/data/Amazon23/${domain}/${domain}.item.json \
        --freeze_LLM False \
        --num_epochs 1
    wait
}

run_rl() {
    cd ${ROOT}
    black rl.py
    accelerate launch \
        --config_file ${ROOT}/config/zero2_opt.yaml \
        --num_processes 2 \
        --main_process_port 29503 \
        rl.py \
        --model_path ${ROOT}/saved_models/${model_name}/Amazon23/${domain}-sft \
        --train_batch_size 128 \
        --eval_batch_size 256 \
        --num_train_epochs 10 \
        --gradient_accumulation_steps 4 \
        --train_file ${dataset_dir}/train/${domain}.csv \
        --eval_file ${dataset_dir}/valid/${domain}.csv \
        --info_file ${dataset_dir}/info/${domain}.txt \
        --category ${domain} \
        --sample_train False \
        --eval_step 0.0999 \
        --reward_type ranking \
        --num_generations 16 \
        --mask_all_zero False \
        --dynamic_sampling False \
        --sync_ref_model True \
        --beam_search True \
        --test_during_training False \
        --temperature 1.0 \
        --learning_rate 1e-5 \
        --add_gt False \
        --beta 1e-3 \
        --dapo False \
        --output_dir ${output_dir} \
        --sid_index_path ${ROOT}/data/Amazon23/${domain}/${domain}.index.json \
        --item_meta_path ${ROOT}/data/Amazon23/${domain}/${domain}.item.json
    wait
}

run_test() {
    cudalist="2 3"
    batch_size=32
    dataset_dir=${ROOT}/datasets/Amazon23/${domain}
    
    cd ${ROOT}

    for exp_name in pretrain sft rl
    do
        if [ "$exp_name" = "pretrain" ]; then
            model_path=/ssd/hust-tangxi/workspace/pretrained-models/${model_name}
        else
            model_path=${ROOT}/saved_models/${model_name}/Amazon23/${domain}-${exp_name}
        fi
        
        test_file=${dataset_dir}/test/${domain}.csv
        info_file=${dataset_dir}/info/${domain}.txt
        
        if [[ ! -f "$test_file" ]] || [[ ! -f "$info_file" ]]; then
            echo "Error: Test or info file not found for domain $domain"
            continue
        fi
        
        temp_dir="${ROOT}/temp/${domain}-${model_name}-${exp_name}"
        mkdir -p "$temp_dir"
        
        python split.py \
            --input_path ${test_file} \
            --output_path ${temp_dir} \
            --cuda_list $(echo $cudalist | tr ' ' ',')
        
        for i in ${cudalist}
        do
            if [[ -f "$temp_dir/${i}.csv" ]]; then
                CUDA_VISIBLE_DEVICES=$i python -u evaluate.py \
                    --base_model "$model_path" \
                    --info_file "$info_file" \
                    --category ${domain} \
                    --test_data_path "$temp_dir/${i}.csv" \
                    --result_json_data "$temp_dir/${i}.json" \
                    --batch_size $batch_size \
                    --num_beams 50 \
                    --max_new_tokens 256 \
                    --guidance_scale 1.0 \
                    --length_penalty 0.0 &
            fi
        done
        wait
        
        output_dir="${ROOT}/results/${domain}/${model_name}-${exp_name}"
        mkdir -p "$output_dir"
        
        actual_cuda_list=$(ls "$temp_dir"/*.json 2>/dev/null | sed 's/.*\///g' | sed 's/\.json//g' | tr '\n' ',' | sed 's/,$//')
        
        python merge.py \
            --input_path $temp_dir \
            --output_path $output_dir/log.json \
            --cuda_list $actual_cuda_list
        wait
        
        python calc.py \
            --path $output_dir/log.json \
            --item_path $info_file \
            --output_path $output_dir/metric.json
        wait

        rm -rf ${temp_dir}
    done
}

# 主函数：根据action参数调用对应函数
main() {
    local action=$1
    
    case $action in
        "prepare")
            run_prepare
            ;;
        "rqvae")
            run_rqvae
            ;;
        "sft")
            run_sft
            ;;
        "rl")
            run_rl
            ;;
        "test")
            run_test
            ;;
        *)
            echo "Usage: $0 {prepare|rqvae|sft|rl|test}"
            exit 1
            ;;
    esac
}

# 调用主函数，传入action参数
main "$@"
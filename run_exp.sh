echo "----------------Running 1k------------------"
python inference_test.py --model ./huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987 --batch 1 --max-new-tokens 1024 --prompt "Hello, how are you?" --device cuda --dtype float16 --warmup-iters 100 --greedy
python inference_test.py --model ./saves/llama3-8b/lora-drop-7b-dkyoon-4k/checkpoint-1400 --batch 1 --max-new-tokens 1024 --prompt "Hello, how are you?" --device cuda --dtype float16  --warmup-iters 100 --greedy
echo "----------------Running 32k------------------"
python inference_test.py --model ./huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987 --batch 1 --max-new-tokens 32000 --prompt "Hello, how are you?" --device cuda --dtype float16 --warmup-iters 100 --greedy
python inference_test.py --model ./saves/llama3-8b/lora-drop-7b-dkyoon-4k/checkpoint-1400 --batch 1 --max-new-tokens 32000 --prompt "Hello, how are you?" --device cuda --dtype float16  --warmup-iters 100 --greedy
echo "----------------Running 64k------------------"
python inference_test.py --model ./huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987 --batch 1 --max-new-tokens 64000 --prompt "Hello, how are you?" --device cuda --dtype float16 --warmup-iters 100 --greedy
python inference_test.py --model ./saves/llama3-8b/lora-drop-7b-dkyoon-4k/checkpoint-1400 --batch 1 --max-new-tokens 64000 --prompt "Hello, how are you?" --device cuda --dtype float16  --warmup-iters 100 --greedy
echo "----------------Running 128k------------------"
python inference_test.py --model ./huggingface/hub/models--Qwen--Qwen2.5-0.5B/snapshots/060db6499f32faf8b98477b0a26969ef7d8b9987 --batch 1 --max-new-tokens 128000 --prompt "Hello, how are you?" --device cuda --dtype float16 --warmup-iters 100 --greedy
python inference_test.py --model ./saves/llama3-8b/lora-drop-7b-dkyoon-4k/checkpoint-1400 --batch 1 --max-new-tokens 128000 --prompt "Hello, how are you?" --device cuda --dtype float16  --warmup-iters 100 --greedy
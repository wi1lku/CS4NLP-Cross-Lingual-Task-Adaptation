source .venv/bin/activate

langs=("Arabic" "Czech" "German" "Galician" "English" "Spanish" "French" "Finnish" "Hindi" "Icelandic" "Italian" "Russian" "Japanese" "Korean" "Polish" "Portuguese" "Turkish" "Indonesian" "Chinese" "Thai")

for lang in "${langs[@]}"; do
    echo "Processing language: $lang"
    python pos/data.py --data_path pos/data/UD_$lang-PUD/*-test.conllu
done

mv pos/data/*/*-split.conllu pos/data/split/
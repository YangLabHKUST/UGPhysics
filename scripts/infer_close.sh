
subjects=(
    'Electrodynamics'
    'Thermodynamics'
    'GeometricalOptics'
    'Relativity'
    'ClassicalElectromagnetism'
    'ClassicalMechanics'
    'WaveOptics'
    'QuantumMechanics'
    'TheoreticalMechanics'
    'AtomicPhysics'
    'SemiconductorPhysics'
    'Solid-StatePhysics'
    'StatisticalMechanics'
) 

models=(
    "gpt-4o-mini-2024-07-18"
)

for subject in "${subjects[@]}"; do
    for model in "${models[@]}"; do 
        echo "Processing subject: $subject \t Using model: $model"
        python codes/infer_close.py --model $model --subject $subject --max_tokens 8192 
        python codes/eval.py --model_path $model --subject $subject 
    done
    python codes/eval.py --model_path $model --subject all 
done
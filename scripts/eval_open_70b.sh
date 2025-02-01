prefix=""
# your path to the model


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
    "Qwen2.5-Math-72B-Instruct"
)

for model in "${models[@]}"; do 
    for subject in "${subjects[@]}"; do
        echo "Processing subject: $subject \t Using model: $model"
        python codes/generate_open.py --model ${prefix}${model} --system "Please reason step by step, and put your final answer within \\boxed{}." --subject $subject --tensor_parallel_size 8
        python codes/eval.py --model_path $model --subject $subject 
    done
    python codes/eval.py --model_path $model --subject all 
done

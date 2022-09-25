
i=0
n=250

p=0.99
python main.py --output-filepath=./models/ssl-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} &
sleep 0.2

p=0.98
python main.py --output-filepath=./models/ssl-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} &
sleep 0.2



# Baselines (no SSL)
for n in 250 1000 4000; do
  for i in {0..5}; do
    wait -n
    python main.py --disable-ssl --output-filepath=./models/only-supervised-${n}-models/id-000${i} --num_labeled_datapoints=${n} &
    sleep 0.2

    wait -n
    python main.py --output-filepath=./models/ssl-resp-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold="resp" &
    sleep 0.2

    wait -n
    python main.py --output-filepath=./models/ssl-neum-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold="neum" &
    sleep 0.2


    wait -n
    p=0.99
    python main.py --output-filepath=./models/ssl-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} &
    sleep 0.2

    wait -n
    p=0.98
    python main.py --output-filepath=./models/ssl-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} &
    sleep 0.2

    wait -n
    p=0.95
    python main.py --output-filepath=./models/ssl-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} &
    sleep 0.2

    wait -n
    p=0.9
    python main.py --output-filepath=./models/ssl-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} &
    sleep 0.2

    wait -n
    p=0.75
    python main.py --output-filepath=./models/ssl-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} &
    sleep 0.2

    # ***********************

    wait -n
    p=0.99
    python main.py --output-filepath=./models/ssl-cauchy-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} --inference-method=cauchy &
    sleep 0.2

    wait -n
    p=0.98
    python main.py --output-filepath=./models/ssl-cauchy-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} --inference-method=cauchy &
    sleep 0.2

    wait -n
    p=0.95
    python main.py --output-filepath=./models/ssl-cauchy-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} --inference-method=cauchy &
    sleep 0.2

    wait -n
    p=0.9
    python main.py --output-filepath=./models/ssl-cauchy-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} --inference-method=cauchy &
    sleep 0.2

    wait -n
    p=0.75
    python main.py --output-filepath=./models/ssl-cauchy-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} --inference-method=cauchy &
    sleep 0.2


#    for p in "0.99" "0.98" "0.95" "0.9" "0.75"; do
#      python main.py --output-filepath=./models/ssl-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p}
#
#      python main.py --output-filepath=./models/ssl-cauchy-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} --inference-method=cauchy
#
#    done
  done
done

wait


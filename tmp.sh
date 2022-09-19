

# Baselines (no SSL)
for i in {0..4}; do
  for n in 250 1000 4000; do
    # start the first 3 trains right away
    python main.py --disable-ssl --output-filepath=./models/only-supervised-${n}-models/id-000${i} --num_labeled_datapoints=${n} &
    sleep 1

    python main.py --output-filepath=./models/ssl-resp-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold="resp" &
    sleep 1

    python main.py --output-filepath=./models/ssl-neum-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold="neum" &
    sleep 1


    # ***********************
    wait -n
    p=0.99
    python main.py --output-filepath=./models/ssl-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} &
    sleep 1

    wait -n
    p=0.98
    python main.py --output-filepath=./models/ssl-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} &
    sleep 1

    wait -n
    p=0.95
    python main.py --output-filepath=./models/ssl-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} &
    sleep 1

    wait -n
    p=0.9
    python main.py --output-filepath=./models/ssl-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} &
    sleep 1

    wait -n
    p=0.75
    python main.py --output-filepath=./models/ssl-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} &
    sleep 1

    # ***********************

    wait -n
    p=0.99
    python main.py --output-filepath=./models/ssl-cauchy-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} --inference-method=cauchy &
    sleep 1

    wait -n
    p=0.98
    python main.py --output-filepath=./models/ssl-cauchy-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} --inference-method=cauchy &
    sleep 1

    wait -n
    p=0.95
    python main.py --output-filepath=./models/ssl-cauchy-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} --inference-method=cauchy &
    sleep 1

    wait -n
    p=0.9
    python main.py --output-filepath=./models/ssl-cauchy-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} --inference-method=cauchy &
    sleep 1

    wait -n
    p=0.75
    python main.py --output-filepath=./models/ssl-cauchy-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} --inference-method=cauchy &
    sleep 1


#    for p in "0.99" "0.98" "0.95" "0.9" "0.75"; do
#      python main.py --output-filepath=./models/ssl-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p}
#      python main.py --output-filepath=./models/ssl-cauchy-perc${p}-${n}-models/id-000${i} --num_labeled_datapoints=${n} --pseudo-label-percentile-threshold=${p} --inference-method=cauchy
#
#    done
  done
done



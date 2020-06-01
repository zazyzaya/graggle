pkill -9 gunicorn
nohup gunicorn -b 0.0.0.0:8050 graggle > /dev/null &

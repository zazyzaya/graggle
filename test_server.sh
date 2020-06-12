pkill -9 gunicorn
gunicorn -b 0.0.0.0:8050 -t 100 graggle &

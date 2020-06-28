docker build ./ -t flask-backend
docker run -d -p 5000:5000 --rm --name flask_backend_container flask-backend:latest
docker build -t pmmoto-image .
docker run --name pmmoto-container -d -v $(pwd):/home/mpitest/pmmoto pmmoto-image 
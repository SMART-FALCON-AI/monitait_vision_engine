# Easy Ocr Service
## Installation
1. Make sure you have installed the latest nvidia cuda on your machine if you want to use GPU.
2. Install Docker and nvidia-docker2 from the official website. [Install](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
3. Clone this repository.
4. Build the docker image by running `docker build -t piai_classification .` in the root directory of this repository.
5. Run docker compose by running `docker-compose up` in the root directory of this repository.

# Usage
## Inference API
go to `http://0.0.0.1:3001/docs` to see the API documentation. there is a swagger UI there.
There is an example of how to use the API in the `tests` folder.


## Test API
```bash
cd tests
python tests.py
```

## Contact
If you have any questions, please contact me at `am.sharifi@aol.com`

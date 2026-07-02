#!/bin/bash

# Log output
exec > /var/log/user-data.log 2>&1

# Update system
sudo yum update -y

# Install Git & Python
sudo yum install -y git python3

# Upgrade pip
python3 -m pip install --upgrade pip

# Clone repository
cd /home/ec2-user
git clone https://github.com/aiisyahnw/MDUAS_ANW07.git

cd MDUAS_ANW07

# Install dependencies
pip3 install -r requirements.txt

# Run Streamlit
nohup streamlit run AWS/AWS_streamlit.py \
    --server.address 0.0.0.0 \
    --server.port 8501 \
    > streamlit.log 2>&1 &
## Environment Setup of Ithemal (https://github.com/ithemal/Ithemal/tree/master)
You first need to install docker and docker-compose.

To build the docker environment, run docker/docker_build.sh.

Once the docker environment is built, connect to it with docker/docker_connect.sh. 
This will drop you into a tmux shell in the container. It will also start a Jupyter notebook exposed on port 8888 with the password ithemal; nothing depends on this except for your own convenience, so feel free to disable exposing the notebook by removing the port forwarding on lines 37 and 38 in docker/docker-compose.yml. The file system in the container is mounted from the local file system, so changes to the file system on the host will propagate to the docker instance, and vice versa. The container will continue running in the background, even if you exit. The container can be stopped with docker/docker_stop.sh from the host machine. To detach from the container while keeping jobs running, use the normal tmux detach command of Control-b d; running docker/docker_connect.sh will drop you back into the same session.


## Install bhive (timing-harness & disassembler)
git clone https://github.com/ithemal/bhive

## Install nanobench (https://github.com/andreas-abel/nanoBench)
git clone https://github.com/andreas-abel/nanoBench.git

The recommended way for using nanoBench is with the wrapper scripts nanoBench.sh (for the user-space variant) and kernel-nanoBench.sh (for the kernel module). The following examples work with both of these scripts. For the kernel module, we also provide a Python wrapper: kernelNanoBench.py.

For obtaining repeatable results, it can help to disable hyper-threading. This can be done with the disable-HT.sh script.


## Code Structure
-> Merge Values/: Code to merge values of all systems (timing-harness & nanoBench) and find statostics.

-> Move_trained_model(Ithemal)/: Commands to evaluate a trained model over different systems so as not to train the model.

-> Transformer/: Code of Transformer creation, set of the data(train, validate, test) and code for Transformer's evaluation.

-> create tensors for transformer/: Create tensors used for Transformers.

Using multiple GPUs
========

When your machine is equipped with multiple GPUs, you often want only use a subset of all available GPUs. To do this, specify a list of computational devices to use in `tg::lego_initialize()`. For example:

```c++
int main() {
    // Using GPU#5 and GPU#6 (assuming you have at least 7 GPUs available)
    lego_initialize(512, {GPU_5, GPU_6});
    
    // ... the rest of your program
}

```

To list all GPUs available on your machine, you can use `nvidia-smi` command.

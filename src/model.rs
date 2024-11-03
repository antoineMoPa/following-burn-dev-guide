use burn::{
    nn::{
        conv::{Conv2d},
        pool::{AdaptiveAvgPool2d},
        Dropout, Linear, Relu,
    },
    prelude::*,
};

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub conv1: Conv2d<B>,
    pub conv2: Conv2d<B>,
    pub pool: AdaptiveAvgPool2d,
    pub dropout: Dropout,
    pub linear1: Linear<B>,
    pub linear2: Linear<B>,
    pub activation: Relu,
}

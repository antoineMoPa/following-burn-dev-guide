mod model;
use crate::model::Model;
use burn::{
    nn::{
        conv::{Conv2dConfig},
        pool::{AdaptiveAvgPool2dConfig},
        DropoutConfig, LinearConfig, Relu,
    },
};
use burn::{
    backend::Wgpu,
    config::Config,
    prelude::Backend,
};
use derive_builder::Builder;

#[derive(Config, Debug, Builder)]
pub struct ModelConfig {
    num_classes: usize,
    hidden_size: usize,
    #[config(default = "0.5")]
    dropout: f64,
}

impl ModelConfig {
    /// Returns the initialized model.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([1, 8], [3, 3]).init(device),
            conv2: Conv2dConfig::new([8, 16], [3, 3]).init(device),
            pool: AdaptiveAvgPool2dConfig::new([8, 8]).init(),
            activation: Relu::new(),
            linear1: LinearConfig::new(16 * 8 * 8, self.hidden_size).init(device),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

fn main() {
    type MyBackend = Wgpu<f32, i32>;

    let device = Default::default();
    let mut config = ModelConfig::new(10, 512).with_dropout(0.5);
    let model = config.init::<MyBackend>(&device);


    println!("{}", config);
}

mod model;
mod data;
mod training;
mod inference;
use std::path::Path;
use crate::training::TrainingConfig;
use crate::model::Model;
use burn::{
    backend::{Autodiff,Wgpu},
    optim::AdamConfig,
    nn::{
        conv::Conv2dConfig,
        pool::AdaptiveAvgPool2dConfig,
        DropoutConfig, LinearConfig, Relu,
    },
    prelude::{Backend,Config},
    data::dataloader::Dataset,
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

// fn main() {
//     type MyBackend = Wgpu<f32, i32>;
//
//     let device = Default::default();
//     let config = ModelConfig::new(10, 512).with_dropout(0.5);
//     let model = config.init::<MyBackend>(&device);
//
//
//     println!("{}", model);
// }

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAutodiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();
    let artifact_dir = "/tmp/guide";

    // Train only if not yet trained
    let model_path = Path::new(artifact_dir).join("model.mpk");
    if !model_path.exists() {
        crate::training::train::<MyAutodiffBackend>(
            artifact_dir,
            TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
            device.clone(),
        );
    }

    crate::inference::infer::<MyBackend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(1)
            .unwrap(),
    );
}

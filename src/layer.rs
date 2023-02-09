use ndarray::{Array1, Array2};
struct Neuron{
    inputs: Array1<f64>,
    weights: Array2<f64>,
    bias: f64,
}

pub struct Layer{
    neurons: Vec<Neuron>,
    output: Array1<f64>,
}

impl Layer{
    pub fn new(input_size: usize, output_size: usize) -> Layer{
        let mut neurons = Vec::new();
        for _ in 0..output_size{
            let weights = Array2::random((input_size, output_size), rand::distributions::Uniform::new(-1.0, 1.0));
            let inputs = Array1::zeros(input_size);
            let bias = 0.0;
            neurons.push(Neuron{inputs, weights, bias});
        }
        let output = Array1::zeros(output_size);
        Layer{neurons, output}
    }

    pub fn train(&mut self, inputs: Array1<f64>, targets: Array1<f64>, learning_rate: f64){
        self.feed_forward(inputs);
        self.back_propagate(targets, learning_rate);
    }
}
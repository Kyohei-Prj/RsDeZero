use std::fmt::Debug;

use ndarray::{Array2, ScalarOperand};
use num::Float;

struct Variable<T: Float> {
    data: Array2<T>,
    input: Array2<T>,
    output: Array2<T>,
}

impl<T> Variable<T> where T: Float {
    fn new(data: Array2<T>) -> Variable<T> {
        Variable { data, input: Array2::<T>::ones((1,1)), output: Array2::<T>::ones((1,1))}
    }
}

trait Function<T: Float> {
    fn call(&mut self, input: &Array2<T>) -> Array2<T>;
    fn forward(&self, input: &Array2<T>) -> Array2<T>;
}

struct Square<T: Float> {
    input: Array2<T>,
    output: Array2<T>,
}

impl<T> Square<T> where T: Float {
    fn new() -> Square<T> {
        Square {input: Array2::<T>::ones((1,1)), output: Array2::<T>::ones((1,1))}
    }
}

impl<T: Float + Debug> Function<T> for Square<T> {
    fn call(&mut self, input: &Array2<T>) -> Array2<T> {
	self.input = input.clone();
	let output = self.forward(input);
	self.output = output.clone();

	output
    }

    fn forward(&self, input: &Array2<T>) -> Array2<T> {
        input.mapv(|x| x * x)
    }
}

struct Exp<T: Float> {
    input: Array2<T>,
    output: Array2<T>,
}

impl<T> Exp<T> where T: Float {
    fn new() -> Exp<T> {
	Exp {input: Array2::<T>::ones((1,1)), output: Array2::<T>::ones((1,1))}
    }
}

impl<T: Float> Function<T> for Exp<T> {
    fn call(&mut self, input: &Array2<T>) -> Array2<T> {
	self.input = input.clone();
	let output = self.forward(input);
	self.output = output.clone();

	output
    }
    fn forward(&self, input: &Array2<T>) -> Array2<T> {
        input.mapv(|x| x.exp())
    }
}

struct ChainedFunc<T: Float> {
    funcs: Vec<Box<dyn Function<T>>>
}

impl<T> ChainedFunc<T> where T: Float + ScalarOperand + Debug{
    fn new() -> ChainedFunc<T>{
	ChainedFunc { funcs: vec![] }
    }
    fn add(&mut self, func: Box<dyn Function<T>>) {
	self.funcs.push(func);
    }
    fn forward(&mut self, input: &Array2<T>) -> Array2<T>{
	let mut output = input.clone();
	for func in &mut self.funcs {
	    output = func.call(&output);
	}

	output
    }
    fn numerical_diff(&mut self, input: &Array2<T>) -> Array2<T>{

	// single epsilon is too small to calculate diff
	// therefore, we doubled the value
	let eps: T = Float::epsilon();
	let eps = eps + eps;

	let x0 = input - eps;
	let x1 = input + eps;
	let y0 = self.forward(&x0);
	let y1 = self.forward(&x1);

	(y1 - y0) / (eps+eps)
    }
    
}

#[cfg(test)]
mod tests {
    use crate::steps::step04::{Function, Square, Variable, Exp, ChainedFunc};
    use assert_approx_eq::assert_approx_eq;
    use ndarray::array;

    #[test]
    fn test_step_4_float() {
        let expected = array![[1.0, 4.0], [9.0, 16.0]];

        let var = Variable::new(array![[1.0, 2.0], [3.0, 4.0]]);
        let square = Square::new();
        let acutal = square.forward(&var.data);
        assert_eq!(acutal, expected);
    }
    #[test]
    fn test_step_4_exp() {
        let expected = array![[0.3678, 0.6065], [1.0, 2.7182]];

        let var = Variable::new(array![[-1.0, -0.5], [0.0, 1.0]]);
        let exp = Exp::new();
        let actual = exp.forward(&var.data);

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_approx_eq!(*a as f64, e, 1e-3f64);
        }
    }
    #[test]
    fn test_step_4_diff() {
	let expected = array![[-32.0, -4.0, -0.5, 0.0], [0.0, 0.5, 4.0, 32.0]];
	
        let var = Variable::new(array![[-2.0, -1.0, -0.5, 0.0], [0.0, 0.5, 1.0, 2.0]]);

	// f(x) = x^4
	let square = Square::new();
	let square2 = Square::new();
	let mut cf = ChainedFunc::new();
	cf.add(Box::new(square));
	cf.add(Box::new(square2));

	// f'(x) = 4x^3
	let actual = cf.numerical_diff(&var.data);

	for (a, e) in actual.iter().zip(expected.iter()) {
            assert_approx_eq!(*a as f64, e, 1e-3f64);
        }
    }
}

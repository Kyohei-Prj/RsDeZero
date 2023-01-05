use ndarray::{Array2, ScalarOperand};
use num::{Float, FromPrimitive, Num, NumCast};

trait FlexFloat: Float + ScalarOperand {}
impl<T> FlexFloat for T where T: Float + ScalarOperand{}

fn cast_float<T, U>(x: T) -> U where T: Num + NumCast + Copy, U: Num + NumCast + Copy {
    U::from(x).unwrap()
}

struct Variable<T: FlexFloat> {
    data: Array2<T>,
    grad: Array2<T>,
}

impl<T: FlexFloat> Variable<T> {
    fn new(data: Array2<T>) -> Variable<T> {
        Variable { data, grad: Array2::<T>::zeros((1, 1))}
    }
}

trait Function<T: FlexFloat> {
    fn forward(&mut self, x: &Array2<T>) -> Variable<T>;
    fn backward(&mut self, dy: &Array2<T>) -> Array2<T>;
}

struct Square<T: FlexFloat> {
    x: Array2<T>,
}

impl<T: FlexFloat> Square<T> {
    fn new() -> Square<T> {
	Square {x: Array2::<T>::zeros((1,1))}
    }
}

impl<T: FlexFloat> Function<T> for Square<T>  {
    fn forward(&mut self, x: &Array2<T>) -> Variable<T> {
	self.x = x.clone();
	let y = x.mapv(|val| val * val);
	let output = Variable::new(y);

	output
    }
    fn backward(&mut self, dy: &Array2<T>) -> Array2<T> {
	let x = &self.x;
	let two: T = cast_float(2.0);
	let dx = x * dy;
	let dx = dx * two;

	dx
    }
    
}

struct Exp<T: FlexFloat> {
    x: Array2<T>,
}

impl<T: FlexFloat> Exp<T> {
    fn new() -> Exp<T> {
	Exp {x: Array2::<T>::ones((1,1))}
    }
}

impl<T: FlexFloat> Function<T> for Exp<T> {
    fn forward(&mut self, x: &Array2<T>) -> Variable<T> {
	self.x = x.clone();
	let y = x.mapv(|val| val.exp());
	let output = Variable::new(y);

	output
    }
    fn backward(&mut self, dy: &Array2<T>) -> Array2<T> {
	let x = &self.x;
	let dx = x.mapv(|val| val.exp()) * dy;

	dx
    }

}


#[cfg(test)]
mod tests {
    use ndarray::{array, Array2};
    use assert_approx_eq::assert_approx_eq;
    use super::{Variable, Square, Exp, Function};

    #[test]
    fn test_step6_square_forward() {
	let expected = array![[4.0, 1.0, 0.0], [0.0, 1.0, 4.0]];

	let input = array![[-2.0, -1.0, 0.0], [0.0, 1.0, 2.0]];
	let var = Variable::new(input);
	let mut square = Square::new();
	let actual = square.forward(&var.data);

	assert_eq!(actual.data, expected);
    }
    #[test]
    fn test_step6_square_backward() {
	let expected = array![[-4.0, -2.0, 0.0], [0.0, 2.0, 4.0]];

	let input = array![[-2.0, -1.0, 0.0], [0.0, 1.0, 2.0]];
	let var = Variable::new(input);
	let mut square = Square::new();
	let _ = square.forward(&var.data);
	let dy = array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]];
	let dx = square.backward(&dy);

	assert_eq!(dx, expected);
    }
    #[test]
    fn test_step_6_exp_forward() {
        let expected = array![[0.3678, 0.6065], [1.0, 2.7182]];

        let var = Variable::new(array![[-1.0, -0.5], [0.0, 1.0]]);
        let mut exp = Exp::new();
        let actual = exp.forward(&var.data).data;

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_approx_eq!(*a as f64, e, 1e-3f64);
        }
    }
    #[test]
    fn test_step_6_exp_backward() {
        let expected = array![[0.3678, 0.6065], [1.0, 2.7182]];

        let var = Variable::new(array![[-1.0, -0.5], [0.0, 1.0]]);	
        let mut exp = Exp::new();
	let _ = exp.forward(&var.data);
	let dy = array![[1.0, 1.0], [1.0, 1.0]];
        let actual = exp.backward(&dy);

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_approx_eq!(*a as f64, e, 1e-3f64);
        }
    }

}

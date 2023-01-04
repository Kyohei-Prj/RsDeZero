use ndarray::Array2;
use num::Float;

struct Variable<T> {
    data: Array2<T>,
}

impl<T> Variable<T> {
    fn new(data: Array2<T>) -> Variable<T> {
        Variable { data }
    }
}

trait Function<T> {
    fn forward(&self, input: &Array2<T>) -> Array2<T>;
}

struct Square {}

impl Square {
    fn new() -> Square {
        Square {}
    }
}

impl<T: std::ops::Mul<Output = T> + Copy> Function<T> for Square {
    fn forward(&self, input: &Array2<T>) -> Array2<T> {
        input.mapv(|x| x * x)
    }
}

struct Exp {}

impl Exp {
    fn new() -> Exp {
        Exp {}
    }
}

impl<T: Float> Function<T> for Exp {
    fn forward(&self, input: &Array2<T>) -> Array2<T> {
        input.mapv(|x| x.exp())
    }
}

#[cfg(test)]
mod tests {
    use crate::steps::step03::{Function, Square, Variable};
    use assert_approx_eq::assert_approx_eq;
    use ndarray::array;

    use super::Exp;
    #[test]
    fn test_step_3_int() {
        let expected = array![[1, 4], [9, 16]];

        let var = Variable::new(array![[1, 2], [3, 4]]);
        let square = Square::new();
        let acutal = square.forward(&var.data);
        assert_eq!(acutal, expected);
    }
    #[test]
    fn test_step_3_float() {
        let expected = array![[1.0, 4.0], [9.0, 16.0]];

        let var = Variable::new(array![[1.0, 2.0], [3.0, 4.0]]);
        let square = Square::new();
        let acutal = square.forward(&var.data);
        assert_eq!(acutal, expected);
    }
    #[test]
    fn test_step_3_exp() {
        let expected = array![[0.3678, 0.6065], [1.0, 2.7182]];

        let var = Variable::new(array![[-1.0, -0.5], [0.0, 1.0]]);
        let exp = Exp::new();
        let actual = exp.forward(&var.data);

        for (a, e) in actual.iter().zip(expected.iter()) {
            assert_approx_eq!(*a as f64, e, 1e-3f64);
        }
    }
}

use ndarray::Array2;

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

#[cfg(test)]
mod tests {
    use crate::steps::step02::{Function, Square, Variable};
    use ndarray::array;
    #[test]
    fn test_step_2_int() {
        let expected = array![[1, 4], [9, 16]];

        let var = Variable::new(array![[1, 2], [3, 4]]);
        let square = Square::new();
        let acutal = square.forward(&var.data);
        assert_eq!(acutal, expected);
    }
    #[test]
    fn test_step_2_float() {
        let expected = array![[1.0, 4.0], [9.0, 16.0]];

        let var = Variable::new(array![[1.0, 2.0], [3.0, 4.0]]);
        let square = Square::new();
        let acutal = square.forward(&var.data);
        assert_eq!(acutal, expected);
    }
}

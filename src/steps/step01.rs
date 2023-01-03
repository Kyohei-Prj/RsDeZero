use ndarray::Array2;

struct Variable<T> {
    data: Array2<T>,
}

impl<T> Variable<T> {
    fn new(data: Array2<T>) -> Variable<T> {
        Variable { data }
    }
}

#[cfg(test)]
mod tests {
    use crate::steps::step01::Variable;
    use ndarray::array;
    #[test]
    fn test_step_1_int() {
        let expected = array![[1, 2], [3, 4]];

        let var = Variable::new(array![[1, 2], [3, 4]]);
        assert_eq!(var.data, expected);
    }
    #[test]
    fn test_step_1_float() {
        let expected = array![[1.0, 2.0], [3.0, 4.0]];

        let var = Variable::new(array![[1.0, 2.0], [3.0, 4.0]]);
        assert_eq!(var.data, expected);
    }
}

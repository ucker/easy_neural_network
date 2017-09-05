#include <iostream>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
using namespace std;
enum Type {
	NONE, ADD, MUL, SIG, RELU, NORM2
};

enum UNIT_FLAG {
	DATA, GRAD, HEIGHT, WIDTH,
};

class Unit {
public:
	Unit() {
		this->data = new double[1];
		this->grad = new double[1];
		this->data[0] = 0.0;
		this->grad[0] = 0.0;
		this->width_dim = 1;
		this->height_dim = 1;
		this->dataCnt = 1;
	}
	Unit(double val, double grad) {
		this->data = new double[1];
		this->grad = new double[1];
		this->data[0] = val;
		this->grad[0] = grad;
		this->width_dim = 1;
		this->height_dim = 1;
		this->dataCnt = 1;
	}
	Unit(double val) : Unit(val, 0.0) {}
	Unit(int height_dim, int width_dim, double* data) {
		this->dataCnt = width_dim * height_dim;
		this->data = new double[this->dataCnt];
		this->grad = new double[this->dataCnt];
		for (int ii = 0; ii < this->dataCnt; ii++) this->data[ii] = data[ii], this->grad[ii] = 0.0;
		this->width_dim = width_dim;
		this->height_dim = height_dim;

	}
	~Unit() {
		delete[] data;
		delete[] grad;
	}
	// x and y begins with 0,
	double getData(int x, int y) const {
		return this->data[getIndex(x, y)];
	}
	void printData() {
		for (int ii = 0; ii < this->dataCnt; ii++) cout << this->data[ii] << ' ';
		cout << '\n';
	}
	void printGrad() {
		for (int ii = 0; ii < this->dataCnt; ii++) cout << this->grad[ii] << ' ';
		cout << '\n';
	}
	// don't use
	double getGrad(int x, int y) const {
		return this->grad[getIndex(x, y)];
	}
	// don't use
	void putData(int x, int y, double d) {
		this->data[getIndex(x, y)] = d;
	}
	// put new data
	// don't use this function
	void putData(int height, int width, double* d) {
		int m_dataCnt = width * height;
		if (this->dataCnt != m_dataCnt) {
			delete[] this->data;
			this->data = new double[m_dataCnt];
			this->dataCnt = m_dataCnt;
		}
		this->width_dim = width;
		this->height_dim = height;
		for (int ii = 0; ii < this->dataCnt; ii++) this->data[ii] = d[ii];
	}
	void putData(double* d) {
		for (int ii = 0; ii < this->dataCnt; ii++) this->data[ii] = d[ii];
	}
	void putGrad(int x, int y, double g) {
		this->grad[getIndex(x, y)] = g;
	}
	// reshape the matrix
	void reshape(int height, int width)
	{
		int m_dataCnt = width * height;
		if (this->dataCnt != m_dataCnt) {
			delete[] this->data;
			delete[] this->grad;
			this->data = new double[m_dataCnt];
			this->grad = new double[m_dataCnt];
			this->dataCnt = m_dataCnt;
		}
		this->width_dim = width;
		this->height_dim = height;
		setZeros();
	}
	void setGradOnes() {
		for (int ii = 0; ii < this->dataCnt; ii++) this->grad[ii] = 1.0;
	}
	
	void addGrad(int x, int y, double ag) {
		this->grad[getIndex(x, y)] += ag;
	}
	// this reloading is almostly same as getData
	double& operator()(int x, int y) {
		return this->data[getIndex(x, y)];
	}
	// this is the same as getGrad
	// flag =
	// DATA : return data
	// GRAD : return gradient
	// WIDTH : return this->width
	// HEIGHT : return this->height
	double& operator()(int x, int y, UNIT_FLAG flag) {
		if (flag == DATA)
			return this->data[getIndex(x, y)];
		else if (flag == GRAD)
			return this->grad[getIndex(x, y)];
	}
	double operator()(UNIT_FLAG flag) const {
		if (flag == HEIGHT)
			return this->height_dim;
		else if (flag == WIDTH)
			return this->width_dim;
	}
	// update weight
	void update(double rate) {
		for (int ii = 0; ii < this->dataCnt; ii++) this->data[ii] -= rate * this->grad[ii];
		setGradZeros();
	}
	void setZeros() {
		for (int ii = 0; ii < dataCnt; ii++) this->data[ii] = 0, this->grad[ii] = 0;
	}
	void setGradZeros() {
		for (int ii = 0; ii < dataCnt; ii++) this->grad[ii] = 0;
	}
	// this function is used for testing
	// computate gradient in two ways
	void addData(int x, int y, double h) {
		this->data[getIndex(x, y)] += h;
	}
	int getWidth() const { return this->width_dim; }
	int getHeight() const { return this->height_dim; }
	int getCnt() const { return this->dataCnt; }
private:
	double* data;
	double* grad;
	int width_dim;
	int height_dim;
	int dataCnt;
	int getIndex(int x, int y) const {
		return x * this->width_dim + y;
	}

};

class Gate {
public:
	Gate() {
	};
	~Gate() {
	};
	virtual void forward() = 0;
	virtual void backward() = 0;
	virtual Type getType() = 0;
	virtual Unit& getData() = 0;
};

class AddGate : public Gate
{
public:
	AddGate() : input1(this->data), input2(this->data) {
		this->unit_type = ADD;
	}
	AddGate(Unit& l, Unit& r) : input1(l), input2(r) {
		this->unit_type = ADD;
	}
	~AddGate() { }
	void forward() override {
		// not handle exception
		// add exception later 
		int m_matrix = this->input1.getHeight();
		int n_matrix = this->input1.getWidth();
		this->data.reshape(m_matrix, n_matrix);
		for (int ii = 0; ii < m_matrix; ii++) {
			for (int jj = 0; jj < n_matrix; jj++) {
				// d[ii * n_matrix + jj] = this->input1(ii, jj) + this->input2(ii, jj);
				this->data(ii, jj) = this->input1(ii, jj) + this->input2(ii, jj);
			}
        }
        this->input1.setGradZeros();
        this->input2.setGradZeros();
	}
	void backward() override {
		int m_matrix = this->input1.getHeight();
		int n_matrix = this->input1.getWidth();
		for (int ii = 0; ii < m_matrix; ii++) {
			for (int jj = 0; jj < n_matrix; jj++) {
				this->input1(ii, jj, GRAD) += this->data(ii, jj, GRAD);
				this->input2(ii, jj, GRAD) += this->data(ii, jj, GRAD);
			}
		}
	}
	Type getType() override { return this->unit_type; }
	Unit& getData() override { return this->data; }
private:
	Unit& input1;
	Unit& input2;
	Unit data;
	Type unit_type;
};

class MulGate : public Gate {
public:
	MulGate() : input1(this->data), input2(this->data) {
		this->unit_type = MUL;
	}
	MulGate(Unit& l, Unit& r) : input1(l), input2(r) {
		this->unit_type = MUL;
	}
	~MulGate() { }
	void forward() override {
		int m_matrix = this->input1.getHeight();
		int n_matrix = this->input2.getWidth();
		int p_matrix = this->input1.getWidth();
		this->data.reshape(m_matrix, n_matrix);
		for (int ii = 0; ii < m_matrix; ii++)
			for (int jj = 0; jj < n_matrix; jj++) {
				for (int kk = 0; kk < p_matrix; kk++) {
					double aaa = this->input1(ii, kk);
					double bbb = this->input2(kk, jj);
					this->data(ii, jj) += aaa * bbb;
				}
            }
        this->input1.setGradZeros();
        this->input2.setGradZeros();
	}
	void backward() override {
		int m_matrix = this->input1(HEIGHT);
		int n_matrix = this->input2(WIDTH);
		int temp_matrix = this->input1(WIDTH);
		for (int ii = 0; ii < m_matrix; ii++) {
			for (int jj = 0; jj < temp_matrix; jj++) {
				for (int kk = 0; kk < n_matrix; kk++) {
					this->input1(ii, jj, GRAD) += this->data(ii, kk, GRAD) *  this->input2(jj, kk);

				}
			}
		}

		for (int ii = 0; ii < temp_matrix; ii++) {
			for (int jj = 0; jj < n_matrix; jj++) {
				for (int kk = 0; kk < m_matrix; kk++)
					this->input2(ii, jj, GRAD) += this->data(kk, jj, GRAD) * this->input1(kk, ii);
			}
		}
	}
	Type getType() override { return this->unit_type; }
	Unit& getData() override { return this->data; }
	// double getGrad(){return this->grad;}
private:
	Unit& input1;
	Unit& input2;
	Unit data;
	Type unit_type;
};


class SigGate : public Gate {
public:
	SigGate() : input1(this->data) {
		this->unit_type = SIG;
	}
	SigGate(Unit& l) : input1(l) {
		this->unit_type = MUL;
	}
	~SigGate() { }
	void forward() override {
		int m_matrix = this->input1.getHeight();
		int n_matrix = this->input1.getWidth();
		this->data.reshape(m_matrix, n_matrix);
		for (int ii = 0; ii < m_matrix; ii++) {
			for (int jj = 0; jj < n_matrix; jj++) {
				this->data(ii, jj) = sigFun(this->input1(ii, jj));
			}
        }
        this->input1.setGradZeros();
	}
	void backward() override {
		int m_matrix = this->input1(HEIGHT);
		int n_matrix = this->input1(WIDTH);
		for (int ii = 0; ii < m_matrix; ii++) {
			for (int jj = 0; jj < n_matrix; jj++) {
				double sigVal = this->data(ii, jj);
				this->input1(ii, jj, GRAD) += this->data(ii, jj, GRAD) * sigVal * (1.0 - sigVal);
			}
		}
	}
	Type getType() { return this->unit_type; }
	Unit& getData() override { return this->data; }
private:
	double sigFun(double x) const { return 1 / (1 + exp(-x)); }
	Unit& input1;
	Unit data;
	Type unit_type;
};

class ReluGate : public Gate {
public:
	ReluGate() : input1(this->data)
	{
		this->unit_type = RELU;
	}
	ReluGate(Unit& l) : input1(l) {
		this->unit_type = RELU;
	}
	~ReluGate() {}
	void forward() override {
		int m_matrix = this->input1.getHeight();
		int n_matrix = this->input1.getWidth();
		this->data.reshape(m_matrix, n_matrix);
		for (int ii = 0; ii < m_matrix; ii++) {
			for (int jj = 0; jj < n_matrix; ii++) {
				this->data(ii, jj) = reluFun(this->input1(ii, jj));
			}
        }
        this->input1.setGradZeros();
	}
	void backward() override {
		int m_matrix = this->input1(HEIGHT);
		int n_matrix = this->input1(WIDTH);
		for (int ii = 0; ii < m_matrix; ii++) {
			for (int jj = 0; jj < n_matrix; jj++)
				this->input1(ii, jj, GRAD) += (this->input1(ii, jj) > 0 ? 1 : 0) * this->data(ii, jj, GRAD);
		}
	}
	Type getType() { return this->unit_type; }
	Unit& getData() override { return this->data; }
private:
	double reluFun(double x) const { return x > 0 ? x : 0; }
	Unit& input1;
	Unit data;
	Type unit_type;
};

class Norm2Gate : public Gate {
public:
	Norm2Gate() : input1(this->data)
	{
		this->unit_type = NORM2;
	}
	Norm2Gate(Unit& l) : input1(l) {
		this->unit_type = NORM2;
	}
	~Norm2Gate() {}
	void forward() override {
		int m_matrix = this->input1(HEIGHT);
		int n_matrix = this->input1(WIDTH);
		this->data.reshape(1, 1);
		for (int ii = 0; ii < m_matrix; ii++) {
			for (int jj = 0; jj < n_matrix; jj++)
			    this->data(0, 0) += this->input1(ii, jj) * this->input1(ii, jj);
		}
        this->input1.setGradZeros();
	}
	void backward() override {
		int m_matrix = this->input1(HEIGHT);
		int n_matrix = this->input1(WIDTH);
		for (int ii = 0; ii < m_matrix; ii++) {
			for (int jj = 0; jj < n_matrix; jj++)
				this->input1(ii, jj, GRAD) += 2 * this->input1(ii, jj) * this->data(0, 0, GRAD);
		}
	}
	Type getType() { return this->unit_type; }
	Unit& getData() override { return this->data; }
private:
	double reluFun(double x) const { return x > 0 ? x : 0; }
	Unit& input1;
	Unit data;
	Type unit_type;
};


// test
int test_square_poly()
{
	Unit x(3);
    Unit b(-2);
    Unit one(1);
    MulGate mul1(x, x);
    MulGate mul2(b, x);
    AddGate add1(mul1.getData(), mul2.getData());
    AddGate add2(add1.getData(), one);
    double last_val = 1000;
    double now_val;
    mul1.forward();
    mul2.forward();
    add1.forward();
    add2.forward();
    now_val = add1.getData().getData(0, 0);
    for (int ii = 0; ii < 400; ii++) {
        add2.getData().setGradOnes();
        add2.backward();
        // add1.getData().setGradOnes();
        add1.backward();
        mul2.backward();
        mul1.backward();
        // x.printGrad();
        x.printData();
        x.update(0.02);
        mul1.forward();
        mul2.forward();
        add1.forward();
        add2.forward();
        // add2.getData().printData();
        
    }
	return 0;
}
int test_mnist() {
	Unit W1;
	Unit x;
	Unit b1;
	Unit W2;
	Unit b2;
	Unit y;
	double learn_rate = 0.02;
	W1.reshape(100, 784);
	b1.reshape(100, 1);
	x.reshape(784, 1);
	W2.reshape(10, 100);
	b2.reshape(10, 1);
	y.reshape(10, 1);
	MulGate layer00(W1, x);
	AddGate layer01(layer00.getData(), b1);
	SigGate layer02(layer01.getData());
	MulGate layer10(W2, layer02.getData());
	AddGate layer11(layer10.getData(), b2);
	SigGate layer12(layer11.getData());
	AddGate loss0(layer12.getData(), y);
	Norm2Gate loss1(loss0.getData());
	// read data
	double input[784];
    double target[10];
    double weight1[100 * 784];
    double weight_b1[100];
    double weight2[10 * 100];
    double weight_b2[10];
    srand((int)time(0) + rand());
    for (int i = 0; i < 784 * 100; i++){
            weight1[i] = rand()%1000 * 0.001 - 0.5;
	}
		
    for (int j = 0; j < 10 * 100; j++){
            weight2[j] = rand()%1000 * 0.001 - 0.5;
    }
    
    for (int j = 0; j < 100; j++){
        weight_b1[j] = rand()%1000 * 0.001 - 0.5;
    }
    for (int k = 0; k < 10; k++){
        weight_b2[k] = rand()%1000 * 0.001 - 0.5;
	}

    W1.putData(weight1);
    W2.putData(weight2);
    b1.putData(weight_b1);
    b2.putData(weight_b2);
	
	FILE *data;
	FILE *index;
	data = fopen("../mnist/tc/train-images.idx3-ubyte", "rb");
	index = fopen("../mnist/tc/train-labels.idx1-ubyte", "rb");
	if (data == NULL || index == NULL) {
		cout << "can't open the file!" << endl;
		exit(0);
	}

	unsigned char image_buf[784];
	unsigned char label_buf[10];

	int useless[1000];
	fread(useless, 1, 16, data);
    fread(useless, 1, 8, index);
	int cnt = 0;
	
	while (!feof(data) && !feof(index)) {
		memset(image_buf, 0, 784);
		memset(label_buf, 0, 10);
		fread(image_buf, 1, 784, data);
		fread(label_buf, 1, 1, index);
		for (int ii = 0; ii < 784; ii++) {
			if ((unsigned int)image_buf[ii] < 128) input[ii] = 0.0;
			else input[ii] = 1.0;
		}
		int label = (unsigned int)label_buf[0];
		for (int ii = 0; ii < 10; ii++) target[ii] = 0.0;
		target[label] = -1.0;
		x.putData(input);
		x.setGradZeros();
		y.putData(target);
		y.setGradZeros();

		layer00.forward();
		layer01.forward();
		layer02.forward();
		layer10.forward();
		layer11.forward();
		layer12.forward();
		loss0.forward();
		loss1.forward();
		loss1.getData().setGradOnes();
		loss1.backward();
		loss0.backward();
		layer12.backward();
		layer11.backward();
		layer10.backward();
		layer02.backward();
		layer01.backward();
		layer00.backward();
		// update weights
		W1.update(0.35);
		b1.update(0.35);
		W2.update(0.35);
		b2.update(0.35);
		cnt++;
		
        if (cnt % 1000 == 0) {
			cout << cnt << "th test is going\n";
			cout << "the loss is ";
			loss1.getData().printData();
        }
	}
	
    fclose(data);
    fclose(index);

	// test
	
	FILE *image_test;
	FILE *image_test_label;
	int test_success_count = 0;
	int test_num = 0;
	image_test = fopen("../mnist/tc/t10k-images.idx3-ubyte", "rb");
	image_test_label = fopen("../mnist/tc/t10k-labels.idx1-ubyte", "rb");
	if (image_test == NULL || image_test_label == NULL){
		cout << "can't open the file!" << endl;
		exit(0);
	}

	fread(useless, 1, 16, image_test);
	fread(useless, 1, 8, image_test_label);

	while (!feof(image_test) && !feof(image_test_label)){
		memset(image_buf, 0, 784);
		memset(label_buf, 0, 10);
		fread(image_buf, 1, 784, image_test);
		fread(label_buf, 1, 1, image_test_label);

		for (int i = 0; i < 784; i++){
			if ((unsigned int)image_buf[i] < 128){
				input[i] = 0.0;
			}
			else{
				input[i] = 1.0;
			}
		}

		for (int k = 0; k < 10; k++){
			target[k] = 0;
		}
		int target_value = (unsigned int)label_buf[0];
		target[target_value] = 1;
		
		x.putData(input);
		layer00.forward();
		layer01.forward();
		layer02.forward();
		layer10.forward();
		layer11.forward();
		layer12.forward();

		double max_value = -99999;
		int max_index = 0;
		for (int k = 0; k < 10; k++){
			if (layer12.getData().getData(k, 0) > max_value){
				max_value = layer12.getData().getData(k, 0);
				max_index = k;
			}
		}

		if (target[max_index] == 1){
			test_success_count ++;
		}
		
		test_num ++;

		if ((int)test_num % 1000 == 0){
			cout << "test num: " << test_num << "  success: " << test_success_count << endl;
		}
	}
	cout << endl;
	cout << "The success rate: " << (double)test_success_count / test_num << endl;
	
    return 0;
}


int test_mnist_1(){
	Unit W1;
	Unit x;
	Unit b1;
	Unit y;
	double learn_rate = 0.02;
	W1.reshape(10, 784);
	b1.reshape(10, 1);
	x.reshape(784, 1);
	y.reshape(10, 1);
	MulGate layer00(W1, x);
	AddGate layer01(layer00.getData(), b1);
	SigGate layer02(layer01.getData());
	AddGate loss0(layer02.getData(), y);
	Norm2Gate loss1(loss0.getData());
	// read data
	double input[784];
    double target[10];
    double weight1[10 * 784];
    double weight_b1[10];
    srand((int)time(0) + rand());
    for (int i = 0; i < 784 * 10; i++){
            weight1[i] = rand()%1000 * 0.001 - 0.5;
	}
		
    
    for (int j = 0; j < 10; j++){
        weight_b1[j] = rand()%1000 * 0.001 - 0.5;
    }

    W1.putData(weight1);
    b1.putData(weight_b1);
	FILE *data;
	FILE *index;
	data = fopen("../mnist/tc/train-images.idx3-ubyte", "rb");
	index = fopen("../mnist/tc/train-labels.idx1-ubyte", "rb");
	if (data == NULL || index == NULL) {
		cout << "can't open the file!" << endl;
		exit(0);
	}

	unsigned char image_buf[784];
	unsigned char label_buf[10];

	int useless[1000];
	fread(useless, 1, 16, data);
    fread(useless, 1, 8, index);
	
	// while (!feof(data) && !feof(index)) {
		memset(image_buf, 0, 784);
		memset(label_buf, 0, 10);
		fread(image_buf, 1, 784, data);
		fread(label_buf, 1, 1, index);
		for (int ii = 0; ii < 784; ii++) {
			if ((unsigned int)image_buf[ii] < 128) input[ii] = 0.0;
			else input[ii] = 1.0;
		}
		int label = (unsigned int)label_buf[0];
		for (int ii = 0; ii < 10; ii++) target[ii] = 0.0;
		target[label] = -1.0;
		x.putData(input);
		y.putData(target);
		W1.setGradZeros();
		b1.setGradZeros();
		x.setGradZeros();
		y.setGradZeros();
        layer00.forward();
		layer01.forward();
		layer02.forward();
		loss0.forward();
		loss1.forward();
		
		loss1.getData().setGradOnes();
		loss1.backward();
		loss0.backward();
		layer02.backward();
		layer01.backward();
		layer00.backward();

	double hh = 0.0001;
	double past_val = loss1.getData().getData(0, 0);
	cout << "The gradient of b computated in computation graph way: \n";
	W1.printGrad();
	cout << "The gradient computated in traditional way: \n";
	for (int ii = 0; ii < W1.getHeight(); ii++) {
		for (int jj = 0; jj < W1.getWidth(); jj++) {
			W1.addData(ii, jj, hh);
			layer00.forward();
			layer01.forward();
			layer02.forward();
		    loss0.forward();
			loss1.forward();
			double now_val = loss1.getData().getData(0, 0);
			cout << now_val / hh - past_val / hh << ' ';
			W1.addData(ii, jj, -1 * hh);
		}
	}
	cout << '\n';
    fclose(data);
	fclose(index);
}
int test_vector() {
	double WW[6] = {1, 2, -3, 4, -1.2, 6};
	double xx[3] = {1.2, 3, -1};
	Unit W(2, 3, WW);
	
	Unit x(3, 1, xx);
	MulGate mul1(W, x);
	Norm2Gate norm(mul1.getData());
	SigGate sig(norm.getData());
	
	
	mul1.forward();
	norm.forward();
	norm.getData().setGradOnes();
	norm.backward();
	mul1.backward();

	double value_before = norm.getData().getData(0, 0);
	cout << "The value computated by graph method is \n";
	x.printGrad();
	cout << "The value compuated by traditional method is \n";
	for (int ii = 0; ii < 3; ii++) {
		for (int jj = 0; jj < 1; jj++) {
			x.addData(ii, jj, 0.001);
			mul1.forward();
			norm.forward();
			cout << norm.getData().getData(0, 0) / 0.001 - value_before /  0.001 << ' ';
			x.addData(ii, jj, -0.001);
		}
	}
	return 0;
}

int bp_like_test() {
	double WW[6] = {2, -2, 1.3, 4, 1.6, -1};
	double bb[3] = {1.1, 1.2, 1.3};
	double xx[3] = {-2, 3, 1};
	double yy[3] = {-1, 0, 0};
	Unit w(2, 3, WW);
	Unit x(3, 1, xx);
	Unit b(3, 1, bb);
	Unit y(3, 1, yy);
	MulGate mul(w, x);
	AddGate add(mul.getData(), b);
	AddGate loss0(add.getData(), y);
	Norm2Gate loss(loss0.getData());

	for (int ii = 0; ii < 200; ii++) {
		mul.forward();
		add.forward();
		loss0.forward();
		loss.forward();

		loss.getData().setGradOnes();
		loss.backward();
		loss0.backward();
		add.backward();
		mul.backward();
		w.update(0.02);
		b.update(0.02);
		loss0.getData().printData();
	}
	return 0;
}

int sigmod_test(){
	double data[6] = {1, 2, -3, 4, -1.2, 6};
	Unit x;
	x.reshape(2, 3);
	x.putData(data);
	SigGate sig(x);
	sig.forward();
	cout << "The result of sig gate is \n";
	sig.getData().printData();
	sig.getData().setGradOnes();
	sig.backward();
	double value_before[2][3];
	for (int ii = 0; ii < 2; ii++)
		for (int jj = 0; jj < 3; jj++)
		    value_before[ii][jj] = sig.getData().getData(ii, jj);
	cout << "graph method\n";
	x.printGrad();
	cout << "trad. method\n";
	for (int ii = 0; ii < 2; ii++){
		for (int jj = 0; jj < 3; jj++) {
	        x.addData(ii, jj, 0.001);
	        sig.forward();
	        
			cout << sig.getData().getData(ii, jj) / 0.001 - value_before[ii][jj] / 0.001 << ' ';
			x.addData(ii, jj, -0.001);
		}
		cout << endl;
    }
	return 0;
}
int main() {
	// test_vector();
	test_mnist();
	// bp_like_test();
	// sigmod_test();
	return 0;
}

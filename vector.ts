import { Matrix } from "./matrix";


class Vector
{
    private _vector: Float64Array;

    get count() {
        return this._vector.length;
    }

    public getAt(i: number): number {
        return this._vector[i];
    }

    public setAt(i: number, value: number) {
        this._vector[i] = value;
    }

    constructor(vector: number[]);
    constructor(vector: Float64Array);
    constructor(length: number);
    constructor(...args: any[]) {
        if (typeof args[0] == "number") {
            let length = <number>args[0];
            if (length <= 0) {
                throw new RangeError("Vector length has to be greater than 0.");
            }
            this._vector = new Float64Array(length);
        }
        else {
            this._vector = new Float64Array(args[0]);
        }
    }

    public static onesVector(length: number): Vector {
        let vector = new Vector(length);
        vector._vector.fill(1);
        return vector;
    }

    public static generateRandom(length: number): Vector;
    public static generateRandom(length: number, minVal: number, maxVal: number): Vector;
    public static generateRandom(length: number, generator: () => number): Vector;
    public static generateRandom(...args: any[]): Vector {
        let length = <number>args[0];

        if (args.length == 2) {
            let generator = <() => number>args[2];
            return new Vector(length).elementWiseChange(e => generator());
        }

        let minVal = -1.0;
        let maxVal = 1.0;

        if (args.length == 3) {
            minVal = <number>args[1];
            maxVal = <number>args[2];
        }

        var range = maxVal - minVal;
        return new Vector(length).elementWiseChange(
            n => range * Math.random() + minVal
        );
    }

    // #region Helpers
    public elementWiseChange(op: (element: number) => number): Vector {
        let newVector = new Vector(this.count);
        for (let i = 0; i < this.count; i++) {
            newVector._vector[i] = op(this._vector[i]);
        }
        return newVector;
    }

    public elementWiseOp(other: Vector, op: (element1: number, element2: number) => number): Vector {
        if (this.count != other.count)
            throw new RangeError(`Vector is ${this.count} and other vector is ${other.count} when they need to be the same.`);

        let newVector = new Vector(this.count);
        for (let i = 0; i < this.count; i++) {
            newVector._vector[i] = op(this._vector[i], other._vector[i]);
        }
        return newVector;
    }

    public elementWiseAggregate(other: Vector, agg: (acc: number, element1: number, element2: number) => number): number {
        if (this.count != other.count)
            throw new RangeError(`Vector is ${this.count} and other vector is ${other.count} when they need to be the same.`);

        let acc = 0;
        for (let i = 0; i < this.count; i++) {
            acc = agg(acc, this._vector[i], other._vector[i]);
        }
        return acc;
    }
    // #endregion

    // #region Operators
    public sum(): number {
        let sum = 0;
        for (let i = 0; i < this.count; i++) {
            sum += this._vector[i];
        }
        return sum;
    }

    public add(num: number): Vector;
    public add(other: Vector): Vector;
    public add(...args: any[]): Vector {
        let arg = args[0]
        if (typeof arg == 'number') {
            return this.elementWiseChange(e => e + arg);
        }
        else {
            let other = <Vector>arg;
            return this.elementWiseOp(other, (e1, e2) => e1 + e2);
        }
    }

    public subtract(num: number): Vector;
    public subtract(other: Vector): Vector;
    public subtract(...args: any[]): Vector {
        let arg = args[0]
        if (typeof arg == 'number') {
            return this.elementWiseChange(e => e - arg);
        }
        else {
            let other = <Vector>arg;
            return this.elementWiseOp(other, (e1, e2) => e1 - e2);
        }
    }

    public multiply(num: number): Vector;
    public multiply(other: Vector): Vector;
    public multiply(...args: any[]): Vector {
        let arg = args[0]
        if (typeof arg == 'number') {
            return this.elementWiseChange(e => e * arg);
        }
        else {
            let other = <Vector>arg;
            return this.elementWiseOp(other, (e1, e2) => e1 * e2);
        }
    }

    public divide(num: number): Vector;
    public divide(other: Vector): Vector;
    public divide(...args: any[]): Vector {
        let arg = args[0]
        if (typeof arg == 'number') {
            return this.elementWiseChange(e => e / arg);
        }
        else {
            let other = <Vector>arg;
            return this.elementWiseOp(other, (e1, e2) => e1 / e2);
        }
    }

    public equals(other: Vector): boolean;
    public equals(other: Vector, epsilon: number): boolean;
    public equals(...args: any[]): boolean {
        let other = <Vector>args[0];
        let epsilon = 0;
        if (args.length == 2) {
            epsilon = <number>args[1];
        }

        if (other == null || other.count != this.count)
            return false;

        for (let e: number = 0; e < this.count; ++e) {
            if (Math.abs(this._vector[e] - other._vector[e]) > epsilon)
                return false;
        }

        return true;
    }
    // #endregion

    // #region VectorFunctions
    // #region Dot
    public dotVec(other: Vector): number {
        if (this.count != other.count)
            throw new RangeError("Both vectors must be of equal length.");
        let product: number = 0;
        for (let i = 0; i < this.count; i++) {
            product += this._vector[i] * other._vector[i];
        }
        return product;
    }

    public dot(other: Matrix, vertical): Vector
    {
        if (this.count != other.numRows) {
            throw new RangeError(`Vector is size ${this.count} and Matrix has ${other.numRows}. The two need to be equal.`);
        }

        let product = new Vector(other.numCols);
        for (let j = 0; j < other.numCols; j++) {
            for (let i = 0; i < other.numRows; i++) {
                product._vector[j] += this._vector[i] * other.getAt(i, j);
            }
        }
        return product;
    }

    public dotToMatrix(other: Vector): Matrix
    {
        let product = new Matrix(this.count, other.count);
        for (let i = 0; i < this.count; i++) {
            let element = this._vector[i];
            for (let j = 0; j < other.count; j++) {
                product.setAt(i, j, element * other._vector[j]);
            }
        }
        return product;
    }
    // #endregion

    // #region Variance
    public covariance(other: Vector): number {
        if (this.count != other.count)
            throw new RangeError(`Vector is ${this.count} and other vector is ${other.count} when they need to be the same.`);

        let sum = 0;
        let otherSum = 0;
        let productSum = 0;
        for (let i = 0; i < this.count; i++) {
            let value = this._vector[i];
            let otherValue = other._vector[i];
            sum += value;
            otherSum += otherValue;
            productSum += value * otherValue;
        }
        return (productSum / this.count) - ((sum * otherSum) / (this.count * this.count));
    }

    public variance(): number {
        let sum = 0;
        for (let i = 0; i < this.count; i++) {
            let val = this._vector[i];
            sum += val * val;
        }
        return sum / this.count;
    }
    // #endregion
    // #endregion

    public toArray(): Float64Array {
        return new Float64Array(this._vector);
    }

    public duplicate(): Vector {
        return new Vector(this._vector);
    }

    public toString(): string {
        return `[${this._vector.join()}]`;
    }
}

export { Vector };

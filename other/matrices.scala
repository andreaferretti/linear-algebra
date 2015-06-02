import breeze.linalg.DenseMatrix
import breeze.stats.distributions.Uniform

val u = Uniform(0, 1)
val m1 = DenseMatrix.rand(1000, 987, u)
val m2 = DenseMatrix.rand(987, 876, u)

m1 * m2 //loads native libraries

def time(f: => Unit) = {
  val start = System.currentTimeMillis
  f
  val end  = System.currentTimeMillis
  end - start
}

time(for (_ <- 1 to 10) { m1 * m2 })
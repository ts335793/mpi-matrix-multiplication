#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <assert.h>
#include <getopt.h>

#include <algorithm>
#include <functional>
#include <fstream>
#include <iostream>
#include <vector>

#include "densematgen.h"

int num_groups(int num_processes, int repl_fact) {
  return num_processes / repl_fact;
}

int process_group(int num_groups, int mpi_rank) {
  return mpi_rank % num_groups;
}

int group_first_row(int num_rows, int num_groups, int group) {
  int base = num_rows / num_groups;
  int rest = num_rows % num_groups;
  return group * base + std::min(group, rest);
}

int group_first_column(int num_columns, int num_groups, int group) {
  return group_first_row(num_columns, num_groups, group);
}

int row_owner_group(int num_rows, int num_groups, int row_num) {
  int base = num_rows / num_groups;
  int rest = num_rows % num_groups;
  if (row_num <= rest * (base + 1)) return row_num / (base + 1);
  else return rest + (row_num - rest * (base + 1)) / base;
}

int column_owner_group(int num_columns, int num_groups, int column_num) {
  return row_owner_group(num_columns, num_groups, column_num);
}

struct sparse_matrix {
  std::vector<int> x;
  std::vector<int> y;
  std::vector<double> v;

  void resize(int size) {
    x.resize(size);
    y.resize(size);
    v.resize(size);
  }

  void sort(bool horizontally) {
    std::vector<int> order(v.size());
    std::vector<int> new_x(x.size());
    std::vector<int> new_y(y.size());
    std::vector<double> new_v(v.size());

    for (int i = 0; i < (int) v.size(); i++) order[i] = i;

    std::sort(order.begin(), order.end(), [this, &horizontally](const int &a, const int &b) {
      if (horizontally) return x[a] < x[b];
      else return y[a] < y[b];
    });

    for (int i = 0; i < (int) order.size(); i++) {
      new_x[i] = x[order[i]];
      new_y[i] = y[order[i]];
      new_v[i] = v[order[i]];
    }

    std::swap(x, new_x);
    std::swap(y, new_y);
    std::swap(v, new_v);
  }

  void scatterv(bool horizontally, int matrix_size, int rank, int root, int comm_size, MPI_Comm &mpi_comm) {
    int size;
    std::vector<int> count;
    std::vector<int> displacements;

    if (rank == root) {
      count.resize(comm_size, 0);
      displacements.resize(comm_size, 0);

      sort(horizontally);

      if (horizontally) for (int i = 0; i < (int) v.size(); i++) count[column_owner_group(matrix_size, comm_size, x[i])]++;
      else for (int i = 0; i < (int) v.size(); i++) count[column_owner_group(matrix_size, comm_size, y[i])]++;

      for (int i = 1; i < (int) displacements.size(); i++) displacements[i] = displacements[i - 1] + count[i - 1];
    }

    MPI_Scatter((void *) count.data(), 1, MPI_INT, (void *) &size, 1, MPI_INT, 0, mpi_comm);

    std::vector<int> new_x(size);
    std::vector<int> new_y(size);
    std::vector<double> new_v(size);

    MPI_Scatterv((void *) x.data(), count.data(), displacements.data(), MPI_INT, (void *) new_x.data(), new_x.size(), MPI_INT, 0, mpi_comm);
    MPI_Scatterv((void *) y.data(), count.data(), displacements.data(), MPI_INT, (void *) new_y.data(), new_y.size(), MPI_INT, 0, mpi_comm);
    MPI_Scatterv((void *) v.data(), count.data(), displacements.data(), MPI_DOUBLE, (void *) new_v.data(), new_v.size(), MPI_DOUBLE, 0, mpi_comm);

    std::swap(x, new_x);
    std::swap(y, new_y);
    std::swap(v, new_v);
  }

  void bcast(int root, MPI_Comm &mpi_comm) {
    int size = v.size();

    MPI_Bcast((void *) &size, 1, MPI_INT, root, mpi_comm);

    resize(size);

    MPI_Bcast((void *) x.data(), x.size(), MPI_INT, root, mpi_comm);
    MPI_Bcast((void *) y.data(), y.size(), MPI_INT, root, mpi_comm);
    MPI_Bcast((void *) v.data(), v.size(), MPI_DOUBLE, root, mpi_comm);
  }

  void send(int destination, MPI_Comm &mpi_comm) {
    int size = v.size();

    MPI_Send((void *) &size, 1, MPI_INT, destination, 0, mpi_comm);
    MPI_Send((void *) x.data(), x.size(), MPI_INT, destination, 0, mpi_comm);
    MPI_Send((void *) y.data(), y.size(), MPI_INT, destination, 0, mpi_comm);
    MPI_Send((void *) v.data(), v.size(), MPI_DOUBLE, destination, 0, mpi_comm);
  }

  void recv(int source, MPI_Comm &mpi_comm) {
    MPI_Status status;
    int size = v.size();

    MPI_Recv((void *) &size, 1, MPI_INT, source, 0, mpi_comm, &status);

    resize(size);

    MPI_Recv((void *) x.data(), x.size(), MPI_INT, source, 0, mpi_comm, &status);
    MPI_Recv((void *) y.data(), y.size(), MPI_INT, source, 0, mpi_comm, &status);
    MPI_Recv((void *) v.data(), v.size(), MPI_DOUBLE, source, 0, mpi_comm, &status);
  }

  void from_csr(std::istream &input, int &matrix_size) {
    int size, pref_sum, _;

    input >> matrix_size >> matrix_size >> size >> _;

    resize(size);

    for (auto &e : v) input >> e;
    for (int i = -1, idx = 0; i < matrix_size; i++) {
      input >> pref_sum;
      while (idx < pref_sum) {
        y[idx] = i;
        idx++;
      }
    }
    for (auto &e : x) input >> e;
  }

  void print(std::ostream &output) {
    for (auto e : x) output << e << " ";
    output << std::endl;
    for (auto e : y) output << e << " ";
    output << std::endl;
    for (auto e : v) output << e << " ";
    output << std::endl;
  }
};

struct dense_matrix {
  std::vector<double> v;

  int size() { return v.size(); }

  void resize(int size, int value) { v.resize(size, value); }

  void zero() { std::fill(v.begin(), v.end(), 0); }

  dense_matrix row(int i, int width) {
    dense_matrix result;
    result.v = std::vector<double>(v.begin() + i * width, v.begin() + (i + 1) * width);
    return result;
  }

  void send(int destination, MPI_Comm &mpi_comm) {
    int size = v.size();

    MPI_Send((void *) &size, 1, MPI_INT, destination, 0, mpi_comm);
    MPI_Send((void *) v.data(), v.size(), MPI_DOUBLE, destination, 0, mpi_comm);
  }

  void recv(int source, MPI_Comm &mpi_comm) {
    MPI_Status status;
    int size = v.size();

    MPI_Recv((void *) &size, 1, MPI_INT, source, 0, mpi_comm, &status);

    v.resize(size);

    MPI_Recv((void *) v.data(), v.size(), MPI_DOUBLE, source, 0, mpi_comm, &status);
  }

  void bcast(int root, MPI_Comm &mpi_comm) {
    int size_ = v.size();

    MPI_Bcast((void *) &size_, 1, MPI_INT, root, mpi_comm);

    v.resize(size_);

    MPI_Bcast((void *) v.data(), v.size(), MPI_DOUBLE, root, mpi_comm);
  }

  void gatherv(int matrix_size, int rank, int root, int comm_size, MPI_Comm &mpi_comm) {
    std::vector<int> counts, displacements;
    std::vector<double> gathered_v;

    if (rank == root) {
      gathered_v.resize(matrix_size);
      displacements.push_back(0);
      for (int i = 0; i < comm_size; i++) {
        counts.push_back(group_first_column(matrix_size, comm_size, i + 1) - group_first_column(matrix_size, comm_size, i));
        displacements.push_back(displacements.back() + counts.back());
      }
    }

    MPI_Gatherv((void *) v.data(), v.size(), MPI_DOUBLE, (void *) gathered_v.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, mpi_comm);

    std::swap(v, gathered_v);
  }

  void generate(int matrix_size, int group, int num_groups, std::function<double(int, int)> generator) {
    v.resize(0);

    for (int y = 0; y < matrix_size; y++)
      for (int x = group_first_column(matrix_size, num_groups, group); x < group_first_column(matrix_size, num_groups, group + 1); x++)
        v.push_back(generator(x, y));
  }

  void print(std::ostream &output, int height, int width) {
    for (int y = 0; y < height; y++) {
      for(int x = 0; x < width; x++)
        output << v[y * width + x] << " ";
      output << std::endl;
    }
  }
};

int main(int argc, char * argv[])
{
  int show_results = 0;
  int use_inner = 0;
  int gen_seed = -1;
  int repl_fact = 1;

  int option = -1;

  double comm_start = 0, comm_end = 0, comp_start = 0, comp_end = 0;
  int num_processes = 1;
  int mpi_rank = 0;
  int exponent = 1;
  double ge_element = 0;
  int count_ge = 0;

  int matrix_size = -1;
  sparse_matrix A;
  dense_matrix B, C;

  int num_groups_l, group_l;
  int num_groups_r, group_r;
  MPI_Comm comm_group_l, comm_group_r, comm_l_leaders, comm_r_leaders, comm_l_ring;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);


  while ((option = getopt(argc, argv, "vis:f:c:e:g:")) != -1) {
    switch (option) {
    case 'v': show_results = 1;
      break;
    case 'i': use_inner = 1;
      break;
    case 'f': if ((mpi_rank) == 0) {
        std::ifstream file(optarg);
        A.from_csr(file, matrix_size);
      }
      break;
    case 'c': repl_fact = atoi(optarg);
      break;
    case 's': gen_seed = atoi(optarg);
      break;
    case 'e': exponent = atoi(optarg);
      break;
    case 'g': count_ge = 1;
      ge_element = atof(optarg);
      break;
    default: fprintf(stderr, "error parsing argument %c exiting\n", option);
      MPI_Finalize();
      return 3;
    }
  }
  if ((gen_seed == -1) || ((mpi_rank == 0) && (matrix_size == -1))) {
    fprintf(stderr, "error: missing seed or sparse matrix file; exiting\n");
    MPI_Finalize();
    return 3;
  }

  comm_start =  MPI_Wtime();

  num_groups_l = num_groups(num_processes, repl_fact);
  num_groups_r = num_groups(num_processes, use_inner ? repl_fact : 1);

  group_l = process_group(num_groups_l, mpi_rank);
  group_r = process_group(num_groups_r, mpi_rank);

  int rank_l = mpi_rank / num_groups_l;
  int rank_r = mpi_rank / num_groups_r;

  int group_r_size = num_processes / num_groups_r;

  MPI_Comm_split(MPI_COMM_WORLD, group_l, rank_l, &comm_group_l);
  MPI_Comm_split(MPI_COMM_WORLD, group_r, rank_r, &comm_group_r);
  MPI_Comm_split(MPI_COMM_WORLD, rank_l == 0 ? 0 : MPI_UNDEFINED, mpi_rank, &comm_l_leaders);
  MPI_Comm_split(MPI_COMM_WORLD, rank_r == 0 ? 0 : MPI_UNDEFINED, mpi_rank, &comm_r_leaders);
  MPI_Comm_split(MPI_COMM_WORLD, rank_l, group_l, &comm_l_ring);

  MPI_Bcast((void *) &matrix_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (mpi_rank < num_groups_l)
    A.scatterv(!use_inner, matrix_size, group_l, 0, num_groups_l, comm_l_leaders);
  A.bcast(0, comm_group_l);

  if (mpi_rank < num_groups_r)
    B.generate(matrix_size, group_r, num_groups_r, [&gen_seed](int x, int y) { return generate_double(gen_seed, y, x); });
  B.bcast(0, comm_group_r);

  C.resize(B.size(), 0);

  int num_iterations = use_inner ? num_groups_l / repl_fact : num_groups_l;
  int shift_l = (rank_l * num_iterations) % num_groups_l;

  if (shift_l != 0) {
    std::vector<int> parity(num_groups_l, -1);
    for (int i = 0; i < (int) parity.size(); i++) {
      bool tf = false;
      for (int j = i; parity[j] == -1; j = (j + shift_l) % num_groups_l) {
        parity[j] = tf;
        tf = !tf;
      }
    }
    sparse_matrix new_A;
    int shift_next = (group_l + shift_l) % num_groups_l;
    int shift_previous = (num_groups_l + group_l - shift_l) % num_groups_l;
    if (parity[group_l]) {
      A.send(shift_next, comm_l_ring);
      new_A.recv(shift_previous, comm_l_ring);
    } else {
      new_A.recv(shift_previous, comm_l_ring);
      A.send(shift_next, comm_l_ring);
    }
    std::swap(A, new_A);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  comm_end = MPI_Wtime();

  comp_start = MPI_Wtime();

  int previous = (group_l + num_groups_l - 1) % num_groups_l;
  int next = (group_l + 1) % num_groups_l;
  int B_width = group_first_column(matrix_size, num_groups_r, group_r + 1) - group_first_column(matrix_size, num_groups_r, group_r);

  sparse_matrix new_A;
  dense_matrix collected_C;
  std::vector<int> counts(group_r_size);
  std::vector<int> shifts(group_r_size);
  std::vector<int> displacements(group_r_size + 1);

  for (int i = 0; i < exponent; i++) {
    int pos = (group_l - i * num_iterations + i * num_groups_l) % num_groups_l;
    for (int i = 0; i < group_r_size; i++) {
      counts[i] = 0;
      for (int diff = 0; diff < num_iterations; diff++) {
        counts[i] += (group_first_row(matrix_size, num_groups_l, pos + 1) - group_first_row(matrix_size, num_groups_l, pos)) * B_width;
        pos = (pos - 1 + num_groups_l) % num_groups_l;
      }
      shifts[i] = group_first_row(matrix_size, num_groups_l, (pos + 1) % num_groups_l);
    }

    displacements[0] = 0;
    int idx = 1;
    for (auto it = counts.rbegin(); it != counts.rend(); it++) {
      displacements[idx] = displacements[idx - 1] + *it;
      idx++;
    }
    std::reverse(displacements.begin(), displacements.end() - 1);

    for (int j = 0; j < num_iterations; j++) {
      for (int k = 0; k < (int) A.v.size(); k++) {
        auto &A_x = A.x[k];
        auto A_y = (A.y[k] - shifts[rank_r] + matrix_size) % matrix_size;
        auto &A_v = A.v[k];
        for (int C_x = 0; C_x < B_width; C_x++) {
          C.v[A_y * B_width + C_x] += A_v * B.v[A_x * B_width + C_x];
        }
      }
      if (num_groups_l > 1) {
        if (group_l % 2) {
          A.send(next, comm_l_ring);
          new_A.recv(previous, comm_l_ring);
        } else {
          new_A.recv(previous, comm_l_ring);
          A.send(next, comm_l_ring);
        }
        std::swap(A, new_A);
      }
    }

    if (rank_r == 0) collected_C.resize(matrix_size * B_width, -1);
    MPI_Gatherv(C.v.data(), counts[rank_r], MPI_DOUBLE, collected_C.v.data(), counts.data(), displacements.data(), MPI_DOUBLE, 0, comm_group_r);

    if (rank_r == 0) {
      int a = (displacements[0] / B_width - shifts[0] + matrix_size) % matrix_size;
      int b = a * B_width;
      std::reverse(collected_C.v.data(), collected_C.v.data() + b);
      std::reverse(collected_C.v.data() + b, collected_C.v.data() + collected_C.v.size());
      std::reverse(collected_C.v.data(), collected_C.v.data() + collected_C.v.size());
      std::swap(C, collected_C);
    }

    C.bcast(0, comm_group_r);
    std::swap(B, C);
    C.zero();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  comp_end = MPI_Wtime();

  if (show_results && mpi_rank < num_groups_r)
  {
    dense_matrix row;
    if (mpi_rank == 0) std::cout << matrix_size << " " << matrix_size << std::endl;
    for (int i = 0; i < matrix_size; i++) {
      row = B.row(i, B_width);
      row.gatherv(matrix_size, mpi_rank, 0, num_groups_r, comm_r_leaders);
      if (mpi_rank == 0) row.print(std::cout, 1, matrix_size);
    }
  }
  if (count_ge)
  {
    int counter = 0;
    int sum = 0;
    for (int y = group_first_row(matrix_size, group_r_size, rank_r); y < group_first_row(matrix_size, group_r_size, rank_r + 1); y++)
      for (int x = 0; x < B_width; x++)
        if (B.v[y * B_width + x] >= ge_element)
          counter++;
    MPI_Reduce((void *) &counter, (void *) &sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (mpi_rank == 0) std::cout << sum << std::endl;
  }

  MPI_Finalize();
  return 0;
}

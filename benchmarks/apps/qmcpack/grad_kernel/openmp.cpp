template<typename GT>
  static void mw_evalGrad(const RefVectorWithLeader<This_t>& engines,
                          const std::vector<const Value*>& dpsiM_row_list,
                          int rowchanged,
                          std::vector<GT>& grad_now)
  {
    auto& engine_leader = engines.getLeader();
    auto& buffer_H2D    = engine_leader.mw_mem_->buffer_H2D;
    auto& grads_value_v = engine_leader.mw_mem_->grads_value_v;

    const int norb = engine_leader.get_psiMinv().rows();
    const int nw   = engines.size();
    buffer_H2D.resize(sizeof(Value*) * 2 * nw);
    Matrix<const Value*> ptr_buffer(reinterpret_cast<const Value**>(buffer_H2D.data()), 2, nw);
    for (int iw = 0; iw < nw; iw++)
    {
      ptr_buffer[0][iw] = engines[iw].get_psiMinv().device_data() + rowchanged * engine_leader.get_psiMinv().cols();
      ptr_buffer[1][iw] = dpsiM_row_list[iw];
    }

    constexpr unsigned DIM = GT::Size;
    grads_value_v.resize(nw, DIM);
    auto* __restrict__ grads_value_v_ptr = grads_value_v.data();
    auto* buffer_H2D_ptr                 = buffer_H2D.data();

    PRAGMA_OFFLOAD("omp target teams distribute num_teams(nw) \
                    map(always, to: buffer_H2D_ptr[:buffer_H2D.size()]) \
                    map(always, from: grads_value_v_ptr[:grads_value_v.size()])")
    for (int iw = 0; iw < nw; iw++)
    {
      const Value* __restrict__ invRow_ptr    = reinterpret_cast<const Value**>(buffer_H2D_ptr)[iw];
      const Value* __restrict__ dpsiM_row_ptr = reinterpret_cast<const Value**>(buffer_H2D_ptr)[nw + iw];
      Value grad_x(0), grad_y(0), grad_z(0);
      PRAGMA_OFFLOAD("omp parallel for reduction(+: grad_x, grad_y, grad_z)")
      for (int iorb = 0; iorb < norb; iorb++)
      {
        grad_x += invRow_ptr[iorb] * dpsiM_row_ptr[iorb * DIM];
        grad_y += invRow_ptr[iorb] * dpsiM_row_ptr[iorb * DIM + 1];
        grad_z += invRow_ptr[iorb] * dpsiM_row_ptr[iorb * DIM + 2];
      }
      grads_value_v_ptr[iw * DIM]     = grad_x;
      grads_value_v_ptr[iw * DIM + 1] = grad_y;
      grads_value_v_ptr[iw * DIM + 2] = grad_z;
    }

    for (int iw = 0; iw < nw; iw++)
      grad_now[iw] = {grads_value_v[iw][0], grads_value_v[iw][1], grads_value_v[iw][2]};
  }
  
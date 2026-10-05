/**
 * (C) Copyright 2020, 2021, 2022, 2023, 2024 IBM. All Rights Reserved.
 *
 * Licensed under the MIT license. See LICENSE file in the project root for details.
 */

#pragma once

#include "math_util.h"
#include "rng.h"
#include "rpu.h"
#include "rpu_pulsed_meta_parameter.h"
#include "rpu_simple_device.h"
#include "utility_functions.h"
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace RPU {

template <typename T> class PulsedRPUDeviceBase;

template <typename T> class PulsedRPUDevice;

template <typename T> struct PulsedRPUDeviceMetaParameterBase : SimpleRPUDeviceMetaParameter<T> {

  PulsedRPUDeviceMetaParameterBase() { this->drift.unsetSimpleDrift(); }

  std::string getName() const override { return "PulsedRPUDeviceParameterBase"; };
  PulsedRPUDeviceBase<T> *createDevice(int x_size, int d_size, RealWorldRNG<T> *rng) override {
    RPU_FATAL("Needs implementation");
  };
  PulsedRPUDeviceMetaParameterBase<T> *clone() const override {
    RPU_FATAL("Needs implementation");
  };
  DeviceUpdateType implements() const override { return DeviceUpdateType::Undefined; };

  virtual T calcWeightGranularity() const { RPU_FATAL("Needs implementation."); };
  virtual T calcNumStates() const { RPU_FATAL("Needs implementation."); };

  friend void
  swap(PulsedRPUDeviceMetaParameterBase<T> &a, PulsedRPUDeviceMetaParameterBase<T> &b) noexcept {
    using std::swap;
    swap(
        static_cast<SimpleRPUDeviceMetaParameter<T> &>(a),
        static_cast<SimpleRPUDeviceMetaParameter<T> &>(b));
  }
};

template <typename T> struct PulsedRPUDeviceMetaParameter : PulsedRPUDeviceMetaParameterBase<T> {

  bool legacy_params = false; // to not load the reset / drift params

  T dw_min = (T)0.001;
  T dw_min_dtod = (T)0.3;
  T dw_min_std = (T)0.3; // ctoc of pulse
  bool dw_min_dtod_log_normal = false;

  T w_min = (T)-0.6;
  T w_min_dtod = (T)0.3;

  T w_max = (T)0.6;
  T w_max_dtod = (T)0.3;

  T up_down = (T)0.0;
  T up_down_dtod = (T)0.01;

  // T lifetime = 0;  from Simple
  T lifetime_dtod = (T)0.0;

  // T diffusion = 0; from Simple
  T diffusion_dtod = (T)0.0;

  bool enforce_consistency = false;
  bool perfect_bias = false;

  T corrupt_devices_prob = (T)0.0;
  T corrupt_devices_range = std::numeric_limits<T>::max();

  T reset = (T)0.0; // mean
  // T reset_std = (T)0.0;  from SimpleDevice
  T reset_dtod = (T)0.0;

  bool adjust_bounds_with_up_down = false;
  T adjust_bounds_with_up_down_dev = (T)0.0;

  T write_noise_std = (T)0.0;
  bool apply_write_noise_on_set = true;
  bool count_pulses = false; // whether to count the pulses. Some runtime penalty

  // Half-select array scheme. hs_mode 0 models a shielded array (DNO opcodes:
  // undriven rows are actively nulled, so half-selected cells feel no pair
  // drive; only the sign-flip decay at coincidences remains). hs_mode 1 models
  // the unshielded array (NORMAL opcodes): every HS state transition of a
  // half-selected cell applies an ordered-pair drive:
  //   HS1<->HS2 (potentiating pair, N1<->N2): +hs_pair_step_up * scale_up
  //   HS3<->HS4 (depressing pair, N3<->N4):   -hs_pair_step_down * scale_down
  //   HS1<->HS3, HS2<->HS4 (reset-like pair): w *= hs_reset_decay
  //   HS1<->HS4, HS2<->HS3 (inert pairs):     no effect
  // Steps are in units of the cell's per-pulse scale (measured 6T1C: ~0.13 of
  // one coincidence per pair). Only active for Halfselected* pulse types with
  // HS tracking enabled.
  int hs_mode = 0;
  T hs_pair_step_up = (T)0.133;
  T hs_pair_step_down = (T)0.144;
  T hs_reset_decay = (T)0.99;
  // Line-state half-select (hs_mode 2: NORMAL opcodes, 3: DNO opcodes). Every
  // cell remembers the last line that pulsed on it (N1 row-POT, N2 col-POT,
  // N3 row-DEP, N4 col-DEP); a change of that line pulls the weight toward the
  // attractor of the ordered pair, w <- A + (w - A) * (1 - hs_rate):
  //   mode 2: N1<->N2 A = hs_attr_up, N3<->N4 A = hs_attr_down,
  //           N1<->N3 / N2<->N4 A = 0 (hs_reset_pairs, default on), others inert
  //   mode 3: rows are always driven (bit 0 -> complementary line), so the
  //           state is N1 / N3 by the row alone and N1<->N3 decays toward 0
  // Acts on half-selected cells at the transition itself (no coincidence
  // needed). Command polarity: setHSPolarity(+1 POT / -1 DEP / 0 infer).
  T hs_rate = (T)0.0;
  T hs_attr_up = (T)1.0;
  T hs_attr_down = (T)-1.0;
  bool hs_reset_pairs = true;

  // Load-dependent step (array interference law, fitted on the 6T1C die-2
  // array, model M3w). A coincidence in a BL slot whose row holds n_row
  // coincidences, whose column holds n_col and with n_other coincidences on
  // neither line has its step (scale and slope) multiplied by
  //   f = 1 / (1 + s * (load_a_row * (n_row-1)^load_p_row
  //                     + load_a_col * (n_col-1) + load_a_other * n_other))
  // with the weak-cell factor s = (mean(scale) / scale_ij)^load_beta taken over
  // the scale of the update direction. load_dno adds the DNO cross cells
  // (inactive row x active column) to the column / other load. Line counts
  // ignore the pulse sign, so drive sign-pure updates to match the hardware.
  // Sparse pulse types only (CPU); implemented by LinearStep.
  bool load_law = false;
  T load_a_row = (T)0.0;
  T load_p_row = (T)1.0;
  T load_a_col = (T)0.0;
  T load_a_other = (T)0.0;
  T load_beta = (T)0.0;
  bool load_dno = false;

  void printToStream(std::stringstream &ss) const override;
  using SimpleMetaParameter<T>::print;
  std::string getName() const override { return "PulsedRPUDeviceParameter"; };
  PulsedRPUDevice<T> *createDevice(int x_size, int d_size, RealWorldRNG<T> *rng) override {
    RPU_FATAL("Needs implementation");
  };
  PulsedRPUDeviceMetaParameter<T> *clone() const override { RPU_FATAL("Needs implementation"); };
  DeviceUpdateType implements() const override { return DeviceUpdateType::Undefined; };

  virtual bool implementsWriteNoise() const { return false; }; // needs to be activated in derived
  virtual bool usesPersistentWeight() const { return write_noise_std > (T)0.0; };
  inline T getScaledWriteNoise() const { return write_noise_std * this->dw_min; };

  void initialize() override {
    PulsedRPUDeviceMetaParameterBase<T>::initialize();
    if (!implementsWriteNoise() && usesPersistentWeight()) {
      RPU_FATAL("Device does not support write noise");
    }
    reset_dtod = MAX(reset_dtod, (T)0.0);
    this->reset_std = MAX(this->reset_std, (T)0.0);
    reset = MAX(reset, (T)0.0);
  };
};

template <typename T> class PulsedRPUDeviceBase : public SimpleRPUDevice<T> {

public:
  // constructor / destructor
  PulsedRPUDeviceBase() {};
  explicit PulsedRPUDeviceBase(int x_sz, int d_sz) : SimpleRPUDevice<T>(x_sz, d_sz) {};
  virtual ~PulsedRPUDeviceBase();

  // NOTE: the HS containers (hs_states_/hs_transition_counts_) are raw pointers
  // owned by this object (freed in the dtor). The defaulted copy/move would
  // shallow-copy those pointers -> double free when a tracking-enabled device is
  // copied (e.g. by .cuda(), which duplicates the CPU device). Implement the
  // rule-of-5 explicitly: copy deep-copies the HS state; move/assign transfer
  // ownership via swap; swap now includes the HS members.
  PulsedRPUDeviceBase(const PulsedRPUDeviceBase<T> &other) : SimpleRPUDevice<T>(other) {
    weight_granularity_ = other.weight_granularity_;
    num_states_ = other.num_states_;
    hs_states_ = nullptr;
    hs_transition_counts_ = nullptr;
    hs_tracking_enabled_ = false;
    hs_polarity_ = other.hs_polarity_;
    hs_last_polarity_ = other.hs_last_polarity_;
    if (other.hs_tracking_enabled_) {
      enableHSTracking(); // allocates fresh containers on this (uses d_size_/x_size_)
      for (int i = 0; i < this->d_size_; ++i) {
        for (int j = 0; j < this->x_size_; ++j) {
          hs_states_[i][j] = other.hs_states_[i][j];
        }
        for (int k = 0; k < 16; ++k) {
          hs_transition_counts_[i][k] = other.hs_transition_counts_[i][k];
        }
      }
    }
  }
  PulsedRPUDeviceBase<T> &operator=(const PulsedRPUDeviceBase<T> &other) {
    PulsedRPUDeviceBase<T> tmp(other);
    swap(*this, tmp);
    return *this;
  }
  PulsedRPUDeviceBase(PulsedRPUDeviceBase<T> &&other) noexcept { swap(*this, other); }
  PulsedRPUDeviceBase<T> &operator=(PulsedRPUDeviceBase<T> &&other) noexcept {
    swap(*this, other);
    return *this;
  }

  friend void swap(PulsedRPUDeviceBase<T> &a, PulsedRPUDeviceBase<T> &b) noexcept {
    using std::swap;
    swap(static_cast<SimpleRPUDevice<T> &>(a), static_cast<SimpleRPUDevice<T> &>(b));
    swap(a.weight_granularity_, b.weight_granularity_);
    swap(a.num_states_, b.num_states_);
    swap(a.hs_states_, b.hs_states_);
    swap(a.hs_transition_counts_, b.hs_transition_counts_);
    swap(a.hs_tracking_enabled_, b.hs_tracking_enabled_);
    swap(a.hs_polarity_, b.hs_polarity_);
    swap(a.hs_last_polarity_, b.hs_last_polarity_);
  }

  virtual void copyInvertDeviceParameter(const PulsedRPUDeviceBase<T> *rpu_device) {
    RPU_FATAL("copyInvertDeviceParameter not available for this device!");
  }

  bool isPulsedDevice() const override { return true; };
  PulsedRPUDeviceBase<T> *clone() const override { RPU_FATAL("Needs implementation"); };

  void
  resetCols(T **weights, int start_col, int n_cols, T reset_prob, RealWorldRNG<T> &rng) override {
    RPU_FATAL("Needs implementation");
  };
  virtual void doSparseUpdate(
      T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng) {
    RPU_FATAL("Sparse update not available for this device!");
  };
  virtual void doSparseUpdateHS(
      T **weights, int i, const int *x_signed_indices, int x_count, int d_sign, RNG<T> *rng) {
    // Default implementation: call standard sparse update
    doSparseUpdate(weights, i, x_signed_indices, x_count, d_sign, rng);
  };
  virtual void doDenseUpdate(T **weights, int *coincidences, RNG<T> *rng) {
    RPU_FATAL("Dense update not available for this device!");
  };
  virtual void doDenseUpdateHS(T **weights, int *coincidences, RNG<T> *rng) {
    // Default implementation: call standard dense update
    doDenseUpdate(weights, coincidences, rng);
  };
  // for Meta-devices [like vector/transfer]: called once before each update starts
  virtual void initUpdateCycle(
      T **weights,
      const PulsedUpdateMetaParameter<T> &up,
      T current_lr,
      int m_batch_info,
      const T *x_input = nullptr,
      const int x_inc = 1,
      const T *d_input = nullptr,
      const int d_inc = 1) {};
  // called when update completed
  virtual void finishUpdateCycle(
      T **weights, const PulsedUpdateMetaParameter<T> &up, T current_lr, int m_batch_info) {};

  inline T getWeightGranularity() const { return weight_granularity_; };
  inline T getNumStates() const { return num_states_; };
  virtual T getPulseCountLearningRate(
      T learning_rate, int current_m_batch, const PulsedUpdateMetaParameter<T> &up) {
    UNUSED(up);
    UNUSED(current_m_batch);
    return learning_rate;
  };

  /* called from the weight updater before the call to
     initUpdateCycle. Can be used to do some additional computation on
     the input */
  virtual void
  initWithUpdateInput(const T *x_input, const int x_inc, const T *d_input, const int d_inc) {};

  void dumpExtra(RPU::state_t &extra, const std::string prefix) override {
    SimpleRPUDevice<T>::dumpExtra(extra, prefix);

    RPU::state_t state;
    RPU::insert(state, "num_states", num_states_);
    RPU::insert(state, "weight_granularity", weight_granularity_);

    RPU::insertWithPrefix(extra, state, prefix);
  };

  void loadExtra(const RPU::state_t &extra, const std::string prefix, bool strict) override {
    SimpleRPUDevice<T>::loadExtra(extra, prefix, strict);

    auto state = RPU::selectWithPrefix(extra, prefix);

    RPU::load(state, "num_states", num_states_, strict);
    RPU::load(state, "weight_granularity", weight_granularity_, strict);
  };

  // Half-selected state management
  inline HalfSelectedState **getHSStates() const { return hs_states_; };
  inline int **getHSTransitionCounts() const { return hs_transition_counts_; };
  inline bool isHSTrackingEnabled() const { return hs_tracking_enabled_; };
  void enableHSTracking();
  void disableHSTracking();
  void resetHSStates();
  void getHSTransitionCounts(std::vector<int> &counts) const;
  void updateHSStateOnly(
      int i, const int *x_signed_indices, int x_count, int d_sign, bool x_pulse_exists, bool d_pulse_exists);
  void updateHSTransitionCount(HalfSelectedState prev_hs, HalfSelectedState curr_hs, int d_idx);
  bool shouldApplyHSDecay(HalfSelectedState prev_hs, HalfSelectedState curr_hs) const;
  // Applies the unshielded (hs_mode 1) ordered-pair drive to a half-selected
  // cell on an HS state transition. No-op by default (devices without bounds/
  // scale containers); PulsedRPUDevice implements it generically.
  virtual void applyHSPairDrive(
      T ** /*weights*/, int /*i*/, int /*j*/, HalfSelectedState /*prev_hs*/,
      HalfSelectedState /*curr_hs*/) {};
  // Load-dependent step (see load_law): prepareLoad is called once per update,
  // setSlotLoad once per BL slot with that slot's active d (row) and x (column)
  // line counts, before the coincidences of the slot are applied.
  // Line-state half-select (hs_mode >= 2): one call per BL slot BEFORE the
  // coincidences of the slot are applied, with that slot's signed pulse indices.
  virtual bool usesHSLineModel() const { return false; };
  virtual void applyHSLineSlot(
      T ** /*weights*/, int /*lr_sign*/, const int * /*d_indices*/, int /*d_count*/,
      const int * /*x_indices_p*/, int /*x_count_p*/, const int * /*x_indices_n*/,
      int /*x_count_n*/) {};
  // state every cell starts from (line-state modes start from "no line yet")
  virtual HalfSelectedState initialHSState() const { return HalfSelectedState::HS1; };
  inline void setHSPolarity(int polarity) { hs_polarity_ = polarity; };
  inline int getHSPolarity() const { return hs_polarity_; };
  virtual bool usesLoadLaw() const { return false; };
  virtual void prepareLoad() {};
  virtual void setSlotLoad(int /*d_count*/, int /*x_count*/) {};

protected:
  inline void setWeightGranularity(T weight_granularity) {
    weight_granularity_ = weight_granularity;
  };
  inline void setNumStates(T num_states) { num_states_ = num_states; };

  void populate(const PulsedRPUDeviceMetaParameterBase<T> &par, RealWorldRNG<T> *rng) {
    SimpleRPUDevice<T>::populate(par, rng);
    setWeightGranularity(par.calcWeightGranularity());
    setNumStates(par.calcNumStates());
  };

  // HS state helper methods (internal use only)
  HalfSelectedState determineHSState(int j_signed, int d_sign) const;
  void allocateHSContainers();
  void freeHSContainers();

protected:
  // Half-selected state tracking
  HalfSelectedState **hs_states_ = nullptr;        // Current HS state for each synapse
  int **hs_transition_counts_ = nullptr;           // 16 transition counters (HS1->HS1, HS1->HS2, etc.)
  bool hs_tracking_enabled_ = false;
  int hs_polarity_ = 0;      // +1 potentiation command, -1 depression, 0 infer from the pulse signs
  int hs_last_polarity_ = 1; // last inferred polarity

private:
  T weight_granularity_ = 0.0;
  T num_states_ = 0.0;
};

template <typename T> class PulsedRPUDevice : public PulsedRPUDeviceBase<T> {

public:
  // constructor / destructor
  PulsedRPUDevice() {};
  /* populate cannot be done through constructor because parameter
     objects reside in derived. Derived populate method needs to
     make sure to call the populate of base class */

  PulsedRPUDevice(int x_size, int d_size);
  ~PulsedRPUDevice();

  PulsedRPUDevice(const PulsedRPUDevice<T> &);
  PulsedRPUDevice<T> &operator=(const PulsedRPUDevice<T> &);
  PulsedRPUDevice(PulsedRPUDevice<T> &&);
  PulsedRPUDevice<T> &operator=(PulsedRPUDevice<T> &&);

  friend void swap(PulsedRPUDevice<T> &a, PulsedRPUDevice<T> &b) noexcept {
    using std::swap;
    swap(static_cast<PulsedRPUDeviceBase<T> &>(a), static_cast<PulsedRPUDeviceBase<T> &>(b));

    swap(a.w_scale_up_, b.w_scale_up_);
    swap(a.w_scale_down_, b.w_scale_down_);
    swap(a.w_max_bound_, b.w_max_bound_);
    swap(a.w_min_bound_, b.w_min_bound_);
    swap(a.w_decay_scale_, b.w_decay_scale_);
    swap(a.w_diffusion_rate_, b.w_diffusion_rate_);
    swap(a.w_persistent_, b.w_persistent_);
    swap(a.w_reset_bias_, b.w_reset_bias_);

    swap(a.containers_allocated_, b.containers_allocated_);
  }

  PulsedRPUDevice<T> *clone() const override { RPU_FATAL("Needs implementation"); };

  void getDPNames(std::vector<std::string> &names) const override;
  void getDeviceParameter(T **weights, std::vector<T *> &data_ptrs) override;
  void setDeviceParameter(T **out_weights, const std::vector<T *> &data_ptrs) override;
  void printDP(int x_count, int d_count) const override;
  int getHiddenWeightsCount() const override;
  void setHiddenWeights(const std::vector<T> &data) override;

  inline T **getPersistentWeights() const { return w_persistent_; };
  inline T **getMaxBound() const { return w_max_bound_; };
  inline T **getMinBound() const { return w_min_bound_; };
  inline T **getDecayScale() const { return w_decay_scale_; };
  inline T **getDiffusionRate() const { return w_diffusion_rate_; };
  inline T **getResetBias() const { return w_reset_bias_; };
  inline T **getScaleUp() const { return w_scale_up_; };
  inline T **getScaleDown() const { return w_scale_down_; };

  void applyHSPairDrive(
      T **weights, int i, int j, HalfSelectedState prev_hs, HalfSelectedState curr_hs) override;

  bool usesHSLineModel() const override { return getPar().hs_mode >= 2; };
  void applyHSLineSlot(
      T **weights, int lr_sign, const int *d_indices, int d_count, const int *x_indices_p,
      int x_count_p, const int *x_indices_n, int x_count_n) override;
  HalfSelectedState initialHSState() const override {
    return getPar().hs_mode >= 2 ? HalfSelectedState::HS0 : HalfSelectedState::HS1;
  };
  bool usesLoadLaw() const override { return getPar().load_law; };
  void prepareLoad() override;
  void setSlotLoad(int d_count, int x_count) override;
  // devices whose sparse update applies loadFactor() override this
  virtual bool supportsLoadLaw() const { return false; };

  PulsedRPUDeviceMetaParameter<T> &getPar() const override {
    return static_cast<PulsedRPUDeviceMetaParameter<T> &>(SimpleRPUDevice<T>::getPar());
  };

  /* Note: In case of persistent data these weight methods below will
     affect the perstistent state and ALL apparent weight elements
     will be re-drawn with noise (even if they were not
     e.g. clipped) */

  void decayWeights(T **weights, bool bias_no_decay) override;
  void decayWeights(T **weights, T alpha, bool bias_no_decay) override;
  void driftWeights(T **weights, T time_since_last_call, RNG<T> &rng) override;
  void diffuseWeights(T **weights, RNG<T> &rng) override;
  void clipWeights(T **weights, T add_clip) override;
  bool onSetWeights(T **weights) override;
  void
  resetCols(T **weights, int start_col, int n_cols, T reset_prob, RealWorldRNG<T> &rng) override;
  virtual void resetAtIndices(T **weights, std::vector<int> x_major_indices, RealWorldRNG<T> &rng);
  void copyInvertDeviceParameter(const PulsedRPUDeviceBase<T> *rpu_device) override;

  using PulsedRPUDeviceBase<T>::dumpExtra;
  using PulsedRPUDeviceBase<T>::loadExtra;

protected:
  void populate(const PulsedRPUDeviceMetaParameter<T> &par, RealWorldRNG<T> *rng);

  T **w_max_bound_ = nullptr;
  T **w_min_bound_ = nullptr;
  T **w_scale_up_ = nullptr;
  T **w_scale_down_ = nullptr;
  T **w_decay_scale_ = nullptr;
  T **w_diffusion_rate_ = nullptr;
  T **w_reset_bias_ = nullptr;
  T **w_persistent_ = nullptr;

  RealWorldRNG<T> write_noise_rng_{0};
  virtual void applyUpdateWriteNoise(T **weights);

  // per-slot load state (rebuilt by prepareLoad / setSlotLoad on every update)
  bool load_active_ = false;
  T load_L_ = (T)0.0;
  std::vector<T> load_s_up_;
  std::vector<T> load_s_down_;
  // step multiplier of cell (i,j) in the current slot; sign > 0 = down (as in the update loops)
  inline T loadFactor(int i, int j, int sign) const {
    int idx = i * this->x_size_ + j;
    T s = sign > 0 ? load_s_down_[idx] : load_s_up_[idx];
    return (T)1.0 / ((T)1.0 + s * load_L_);
  };

private:
  void freeContainers();
  void allocateContainers();
  void initialize();
  bool containers_allocated_ = false;
};

#define BUILD_PULSED_DEVICE_CONSTRUCTORS(                                                          \
    CLASSNAME, CTOR_BODY, DTOR_BODY, COPY_BODY, MOVE_BODY, SWAP_BODY, DPNAMES_BODY, DP2V_BODY,     \
    V2DP_BODY, INVERT_COPY_BODY)                                                                   \
public:                                                                                            \
  explicit CLASSNAME(){};                                                                          \
  explicit CLASSNAME(int x_size, int d_size) : PulsedRPUDevice<T>(x_size, d_size) {                \
    initialize();                                                                                  \
  };                                                                                               \
  explicit CLASSNAME(                                                                              \
      int x_size, int d_size, const CLASSNAME##MetaParameter<T> &par, RealWorldRNG<T> *rng)        \
      : PulsedRPUDevice<T>(x_size, d_size) {                                                       \
    initialize();                                                                                  \
    populate(par, rng);                                                                            \
  };                                                                                               \
  ~CLASSNAME() {                                                                                   \
    if (initialized_) {                                                                            \
      DTOR_BODY;                                                                                   \
    }                                                                                              \
  };                                                                                               \
  CLASSNAME(const CLASSNAME<T> &other) : PulsedRPUDevice<T>(other) {                               \
    if (other.initialized_) {                                                                      \
      initialize();                                                                                \
      COPY_BODY;                                                                                   \
    }                                                                                              \
  };                                                                                               \
  CLASSNAME<T> &operator=(const CLASSNAME<T> &other) {                                             \
    CLASSNAME<T> tmp(other);                                                                       \
    swap(*this, tmp);                                                                              \
    return *this;                                                                                  \
  };                                                                                               \
  CLASSNAME(CLASSNAME<T> &&other) { *this = std::move(other); };                                   \
  CLASSNAME<T> &operator=(CLASSNAME<T> &&other) {                                                  \
    PulsedRPUDevice<T>::operator=(std::move(other));                                               \
    initialized_ = other.initialized_;                                                             \
    MOVE_BODY;                                                                                     \
    return *this;                                                                                  \
  };                                                                                               \
  friend void swap(CLASSNAME<T> &a, CLASSNAME<T> &b) noexcept {                                    \
    using std::swap;                                                                               \
    swap(static_cast<PulsedRPUDevice<T> &>(a), static_cast<PulsedRPUDevice<T> &>(b));              \
    swap(a.initialized_, b.initialized_);                                                          \
    SWAP_BODY;                                                                                     \
  };                                                                                               \
                                                                                                   \
  void copyInvertDeviceParameter(const PulsedRPUDeviceBase<T> *rpu_device) override {              \
    PulsedRPUDevice<T>::copyInvertDeviceParameter(rpu_device);                                     \
    const auto *rpu = dynamic_cast<const CLASSNAME<T> *>(rpu_device);                              \
    if (rpu == nullptr) {                                                                          \
      RPU_FATAL("Wrong device class");                                                             \
    };                                                                                             \
    INVERT_COPY_BODY;                                                                              \
  };                                                                                               \
                                                                                                   \
  void printToStream(std::stringstream &ss) const override {                                       \
    ss << "Device:" << std::endl;                                                                  \
    getPar().printToStream(ss);                                                                    \
  };                                                                                               \
  CLASSNAME<T> *clone() const override { return new CLASSNAME<T>(*this); };                        \
                                                                                                   \
private:                                                                                           \
  void initialize() {                                                                              \
    if (!initialized_) {                                                                           \
      CTOR_BODY;                                                                                   \
      initialized_ = true;                                                                         \
    };                                                                                             \
  };                                                                                               \
  bool initialized_ = false;                                                                       \
                                                                                                   \
protected:                                                                                         \
  void populate(const CLASSNAME##MetaParameter<T> &par, RealWorldRNG<T> *rng);                     \
                                                                                                   \
public:                                                                                            \
  CLASSNAME##MetaParameter<T> &getPar() const override {                                           \
    return static_cast<CLASSNAME##MetaParameter<T> &>(SimpleRPUDevice<T>::getPar());               \
  };                                                                                               \
                                                                                                   \
  void getDPNames(std::vector<std::string> &names) const override {                                \
    PulsedRPUDevice<T>::getDPNames(names);                                                         \
    {DPNAMES_BODY};                                                                                \
  };                                                                                               \
                                                                                                   \
  void getDeviceParameter(T **weights, std::vector<T *> &data_ptrs) override {                     \
                                                                                                   \
    PulsedRPUDevice<T>::getDeviceParameter(weights, data_ptrs);                                    \
                                                                                                   \
    std::vector<std::string> names;                                                                \
    std::vector<std::string> all_names;                                                            \
    this->getDPNames(all_names);                                                                   \
    if (all_names.size() != data_ptrs.size()) {                                                    \
      RPU_FATAL("Wrong number of arguments.");                                                     \
    }                                                                                              \
    PulsedRPUDevice<T>::getDPNames(names);                                                         \
    {DP2V_BODY};                                                                                   \
  };                                                                                               \
                                                                                                   \
  void setDeviceParameter(T **out_weights, const std::vector<T *> &data_ptrs) override {           \
    PulsedRPUDevice<T>::setDeviceParameter(out_weights, data_ptrs);                                \
                                                                                                   \
    std::vector<std::string> names;                                                                \
    PulsedRPUDevice<T>::getDPNames(names);                                                         \
    {V2DP_BODY};                                                                                   \
    this->onSetWeights(out_weights);                                                               \
  }

#define BUILD_PULSED_DEVICE_META_PARAMETER(                                                        \
    DEVICENAME, IMPLEMENTS, PAR_BODY, PRINT_BODY, GRANULARITY_BODY, ADD)                           \
  template <typename T>                                                                            \
  struct DEVICENAME##RPUDeviceMetaParameter : public PulsedRPUDeviceMetaParameter<T> {             \
    PAR_BODY                                                                                       \
                                                                                                   \
    std::string getName() const override { return #DEVICENAME; };                                  \
    DeviceUpdateType implements() const override { return IMPLEMENTS; };                           \
                                                                                                   \
    DEVICENAME##RPUDevice<T> *                                                                     \
    createDevice(int x_size, int d_size, RealWorldRNG<T> *rng) override {                          \
      return new DEVICENAME##RPUDevice<T>(x_size, d_size, *this, rng);                             \
    };                                                                                             \
                                                                                                   \
    DEVICENAME##RPUDeviceMetaParameter<T> *clone() const override {                                \
      return new DEVICENAME##RPUDeviceMetaParameter<T>(*this);                                     \
    };                                                                                             \
                                                                                                   \
    T calcWeightGranularity() const override { GRANULARITY_BODY };                                 \
    T calcNumStates() const override {                                                             \
      return (this->w_max - this->w_min) / calcWeightGranularity();                                \
    };                                                                                             \
                                                                                                   \
    void printToStream(std::stringstream &ss) const override {                                     \
      PulsedRPUDeviceMetaParameter<T>::printToStream(ss);                                          \
      PRINT_BODY                                                                                   \
    };                                                                                             \
                                                                                                   \
    ADD                                                                                            \
  }

// some macros for the Update_W function
#define PULSED_UPDATE_W_LOOP(BODY)                                                                 \
  PRAGMA_SIMD                                                                                      \
  for (int jj = 0; jj < x_count; jj++) {                                                           \
    int j_signed = x_signed_indices[jj];                                                           \
    int sign = (j_signed < 0) ? -d_sign : d_sign;                                                  \
    int j = (j_signed < 0) ? -j_signed - 1 : j_signed - 1;                                         \
    {                                                                                              \
      BODY;                                                                                        \
    }                                                                                              \
  }

#define PULSED_UPDATE_W_LOOP_DENSE(BODY)                                                           \
  int _total_size = this->x_size_ * this->d_size_;                                                 \
  for (int j = 0; j < _total_size; j++) {                                                          \
    int c_signed = coincidences[j];                                                                \
    if (c_signed == 0) {                                                                           \
      continue;                                                                                    \
    }                                                                                              \
    int ac = abs(c_signed);                                                                        \
    int sign = c_signed > 0 ? 1 : -1;                                                              \
    PRAGMA_SIMD                                                                                    \
    for (int i_c = 0; i_c < ac; i_c++) {                                                           \
      BODY;                                                                                        \
    }                                                                                              \
  }

// Macro for HS-aware weight updates
#define PULSED_UPDATE_W_LOOP_HS(BODY, HS_BODY)                                                     \
  PRAGMA_SIMD                                                                                      \
  for (int jj = 0; jj < x_count; jj++) {                                                           \
    int j_signed = x_signed_indices[jj];                                                           \
    int sign = (j_signed < 0) ? -d_sign : d_sign;                                                  \
    int j = (j_signed < 0) ? -j_signed - 1 : j_signed - 1;                                         \
    if (this->hs_tracking_enabled_) {                                                              \
      int d_idx = i;                                                                               \
      HalfSelectedState prev_hs = this->hs_states_[d_idx][j];                                      \
      HalfSelectedState curr_hs = this->determineHSState(j_signed, d_sign);                        \
      this->updateHSTransitionCount(prev_hs, curr_hs, d_idx);                                      \
      this->hs_states_[d_idx][j] = curr_hs;                                                        \
      { HS_BODY; }                                                                                 \
    } else {                                                                                       \
      { BODY; }                                                                                    \
    }                                                                                              \
  }

// Helper macro to determine HS state from pulse pattern
#define DETERMINE_HS_STATE(x_pulse_exists, d_pulse_exists, x_sign, d_sign)                         \
  (x_pulse_exists && d_pulse_exists) ? HalfSelectedState::HS0 :                                    \
  (x_pulse_exists && !d_pulse_exists && x_sign == d_sign) ? HalfSelectedState::HS1 :              \
  (!x_pulse_exists && d_pulse_exists && x_sign == d_sign) ? HalfSelectedState::HS2 :              \
  (x_pulse_exists && !d_pulse_exists && x_sign != d_sign) ? HalfSelectedState::HS3 :              \
  (!x_pulse_exists && d_pulse_exists && x_sign != d_sign) ? HalfSelectedState::HS4 :              \
  HalfSelectedState::HS0

} // namespace RPU

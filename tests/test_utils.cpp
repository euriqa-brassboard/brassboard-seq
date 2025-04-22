//

#include <vector>
#include <sstream>

namespace brassboard_seq {

static rtval::TagVal test_callback_extern(auto *self)
{
    auto res = py::ptr(self->cb)();
    return rtval::TagVal::from_py(res.get());
}

static rtval::TagVal test_callback_extern_age(auto *self, unsigned age)
{
    auto res = py::ptr(self->cb)(py::new_int(age));
    return rtval::TagVal::from_py(res.get());
}

static inline std::vector<int> _get_suffix_array(std::vector<int> S)
{
    int N = S.size();
    std::vector<int> SA(N);
    std::vector<int> ws(N);
    get_suffix_array(SA, S, ws);
    return SA;
}
static inline std::vector<int> _get_height_array(std::vector<int> S,
                                                 std::vector<int> SA)
{
    int N = S.size();
    std::vector<int> RK(N);
    std::vector<int> height(N <= 2 ? 0 : N - 2);
    order_to_rank(RK, SA);
    get_height_array(height, S, SA, RK);
    return height;
}
struct MaxRange {
    int i0;
    int i1;
    int maxv;
};
static inline std::vector<MaxRange> _get_max_range(std::vector<int> value)
{
    std::vector<MaxRange> res;
    foreach_max_range(value, [&] (int i0, int i1, int maxv) {
        res.push_back({ i0, i1, maxv });
    });
    return res;
}

using _Bits_i32x5 = Bits<int32_t,5>;
using _Bits_i64x4 = Bits<int64_t,4>;
using _Bits_u64x4 = Bits<uint64_t,4>;
using _Bits_i8x43 = Bits<int8_t,43>;

template<typename T>
T get_mask(unsigned b1, unsigned b2)
{
    return T::mask(b1, b2);
}

template<typename Res>
auto convert(const auto &v)
{
    return Res(v);
}

auto op_not(const auto &v)
{
    return ~v;
}
template<typename Res>
auto op_or(const auto &v1, const auto &v2)
{
    return Res(v1 | v2);
}
template<typename Res>
auto op_and(const auto &v1, const auto &v2)
{
    return Res(v1 & v2);
}
template<typename Res>
auto op_xor(const auto &v1, const auto &v2)
{
    return Res(v1 ^ v2);
}
void op_ior(auto &v1, const auto &v2)
{
    v1 |= v2;
}
void op_iand(auto &v1, const auto &v2)
{
    v1 &= v2;
}
void op_ixor(auto &v1, const auto &v2)
{
    v1 ^= v2;
}

bool bits_to_bool(const auto &v)
{
    return bool(v);
}

auto *bits_to_pylong(const auto &v)
{
    return v.to_pylong();
}

auto *bits_to_pybytes(const auto &v)
{
    return v.to_pybytes();
}

std::string bits_to_str(const auto &v, bool showbase)
{
    std::ostringstream io;
    if (showbase)
        io << std::showbase;
    io << v;
    return io.str();
}

class test_istream : public std::istream {
public:
    test_istream() : std::istream(&m_buf)
    {}

private:
    pybytes_streambuf m_buf;
};

class test_istream_ba : public std::istream {
public:
    test_istream_ba() : std::istream(&m_buf)
    {}

private:
    pybytearray_streambuf m_buf;
};

static inline auto *_new_time_manager()
{
    return event_time::TimeManager::alloc();
}

static inline auto *timemanager_new_round_time(auto self, auto prev, auto offset,
                                               auto cond, auto wait_for)
{
    return self->new_round(prev, offset, cond, wait_for).rel();
}

static inline auto *timemanager_new_time_int(auto self, auto prev, auto offset,
                                             auto floating, auto cond, auto wait_for)
{
    return self->new_int(prev, offset, floating, cond, wait_for).rel();
}

static inline auto *timemanager_get_event_times(auto self)
{
    return py::newref(self->event_times);
}

static inline auto *condseq_get_cond(py::ptr<> condseq)
{
    if (auto cond = py::exact_cast<seq::ConditionalWrapper>(condseq))
        return py::newref(cond->cond);
    return py::newref(py::arg_cast<seq::TimeSeq>(condseq, "s")->cond);
}

void init_action_obj(auto *action, py::ptr<> value, py::ptr<> cond, bool is_pulse,
                     bool exact_time, py::dict kws, int aid)
{
    py::dict_ref _kws;
    if (!kws.is_none())
        _kws.assign(kws);
    auto p = new action::Action(value, cond, is_pulse, exact_time, std::move(_kws), aid);
    action->action = p;
    action->tofree.reset(p);
}

PyObject *_action_get_cond(action::Action *action)
{
    return py::newref(action->cond);
}

PyObject *_action_get_value(action::Action *action)
{
    return py::newref(action->value);
}

PyObject *_action_get_length(action::Action *action)
{
    if (!action->length)
        Py_RETURN_NONE;
    return py::newref(action->length);
}

PyObject *_action_get_end_val(action::Action *action)
{
    if (!action->end_val)
        Py_RETURN_NONE;
    return py::newref(action->end_val);
}

PyObject *_action_py_str(action::Action *action)
{
    return action->py_str().rel();
}

__attribute__((returns_nonnull))
static inline event_time::TimeManager *seq_get_time_mgr(py::ptr<seq::Seq> seq)
{
    return py::newref(seq->seqinfo->time_mgr);
}

auto *compiledseq_get_all_actions(backend::CompiledSeq &cseq)
{
    return cseq.all_actions.get();
}
auto compiledseq_get_total_time(backend::CompiledSeq &cseq)
{
    return cseq.total_time;
}
void _timemanager_finalize(auto *self)
{
    self->finalize();
}
auto _timemanager_compute_all_times(auto *self, unsigned age)
{
    return self->compute_all_times(age);
}
auto *new_time_rt(auto self, auto prev, auto offset, auto cond, auto wait_for)
{
    return self->new_rt(prev, offset, cond, wait_for).rel();
}
void event_time_set_base_int(auto self, auto base, int64_t offset)
{
    self->set_base_int(base, offset);
}
void event_time_set_base_rt(auto self, auto base, auto offset)
{
    self->set_base_rt(base, offset);
}

}

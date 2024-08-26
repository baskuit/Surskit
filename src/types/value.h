#pragma once

#include <types/rational.h>

#include <gmpxx.h>

template <typename Real>
struct PairReal
{
    static constexpr bool IS_CONSTANT_SUM{false};

    Real row_value{Rational<>{0}};
    Real col_value{Rational<>{0}};

    PairReal() {}
    constexpr PairReal(Real row_value, Real col_value) : row_value{row_value}, col_value{col_value} {}
    template <typename ValueType>
    constexpr PairReal(const ValueType &other) : row_value{other.get_row_value()}, col_value{other.get_col_value()} {}

    inline constexpr Real get_row_value() const
    {
        return row_value;
    }
    inline constexpr Real get_col_value() const
    {
        return col_value;
    }

    PairReal &operator+=(const PairReal &other)
    {
        row_value += other.row_value;
        col_value += other.col_value;
        return *this;
    }

    constexpr PairReal operator+(const PairReal other) const
    {
        return PairReal{Real{row_value + other.row_value}, Real{col_value + other.col_value}};
    }

    constexpr PairReal operator*(const Real val) const
    {
        return PairReal{Real{row_value * val}, Real{col_value * val}};
        // arithmetic types must be explicitly converted to Real members
    }

    friend std::ostream &operator<<(std::ostream &os, const PairReal &value)
    {
        os << value.row_value << ',' << value.col_value;
        return os;
    }
};

template <>
struct PairReal<mpq_class>
{
    static constexpr bool IS_CONSTANT_SUM{false};

    mpq_class x{};
    int y;
    mpq_class row_value{Rational<>{0}};
    mpq_class col_value{Rational<>{0}};

    PairReal() {}
    PairReal(mpq_class row_value, mpq_class col_value) : row_value{row_value}, col_value{col_value} {}
    template <typename ValueType>
    PairReal(const ValueType &other) : row_value{other.get_row_value()}, col_value{other.get_col_value()} {}

    inline mpq_class get_row_value() const
    {
        return row_value;
    }
    inline mpq_class get_col_value() const
    {
        return col_value;
    }

    PairReal<mpq_class> &operator+=(const PairReal<mpq_class> &other)
    {
        row_value += other.row_value;
        col_value += other.col_value;
        row_value.canonicalize(); // TODO TODO add to other functions?
        col_value.canonicalize();
        return *this;
    }

    PairReal<mpq_class> operator+(const PairReal<mpq_class> other) const
    {
        return PairReal<mpq_class>{row_value + other.row_value, col_value + other.col_value};
    }

    PairReal<mpq_class> operator*(const mpq_class val) const
    {
        return PairReal<mpq_class>{static_cast<mpq_class>(row_value * val), static_cast<mpq_class>(col_value * val)};
    }

    friend std::ostream &operator<<(std::ostream &os, const PairReal<mpq_class> &value)
    {
        os << value.row_value.get_str() << ',' << value.col_value.get_str();
        return os;
    }
};

template <int PAYOFF_SUM_NUM, int PAYOFF_SUM_DEN>
struct ConstantSum
{
    template <typename Real>
    struct Value
    {
        static constexpr bool IS_CONSTANT_SUM{true};
        static constexpr Rational<> PAYOFF_SUM{PAYOFF_SUM_NUM, PAYOFF_SUM_DEN};

        Real row_value{Rational<>{0}};

        constexpr Value() {}
        constexpr Value(const Real row_value) : row_value{row_value} {}

        inline constexpr Real get_row_value() const
        {
            return row_value;
        }
        inline constexpr Real get_col_value() const
        {
            return Real{Real{PAYOFF_SUM} - row_value};
        }

        template <typename T>
        Value &operator+=(const T t)
        {
            row_value += static_cast<Real>(t);
            return *this;
        }

        Value &operator+=(const Value t)
        {
            row_value += static_cast<Real>(t.row_value);
            return *this;
        }

        template <typename T>
        Value operator*(const T t)
        {
            return Value{Real{row_value * static_cast<Real>(t)}};
        }

        friend std::ostream &operator<<(std::ostream &os, const Value &x)
        {
            if constexpr (std::is_same_v<Real, mpq_class>)
            {
                os << x.row_value.get_str();
            }
            else
            {
                os << x.row_value;
            }
            return os;
        }
    };
};

template <typename Types>
typename Types::Value make_draw()
{
    if constexpr (Types::Value::IS_CONSTANT_SUM == true)
    {
        return {typename Types::Q{Types::Value::PAYOFF_SUM.p, Types::Value::PAYOFF_SUM.q * 2}};
    }
    else
    {
        return {typename Types::Q{1, 2}, typename Types::Q{1, 2}};
    }
}

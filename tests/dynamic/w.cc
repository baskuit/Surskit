#include <pinyon.h>

/*

TODO: The whole thing. W::Types was a bit of an exericise and after-thought.
It's not as tested or supported as other Pinyon utilities.

Basic check of consistency - getting the same results whether we use W::Types wrapper or the raw class

*/

// Search Types
template <typename Types>
void foo (
    const typename Types::PRNG &device,
    const typename Types::State &state,
    const typename Types::Model &model,
    const typename Types::Search &search
) {
    const size_t iterations = 1 << 10;

    W::Types::PRNG w_device{device.get_seed()};
    auto w_state = W::make_model<Types>(state);
    auto w_model = W::make_model<Types>(model);
    auto w_search = W::make_model<Types>(search);
    auto w_node = W::make_node<Types>();

    typename Types::VectorReal r,c;

    // regular search
    {
        w_search.run_for_iterations(iterations, w_device, w_state, w_model, w_node);
    };

    // w search
    {
        
    };

    assert(true);
}

int main () {

};
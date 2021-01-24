//
// Created by Dekai WU and YAN Yuchen on 20200503.
//

#ifndef LEGO_EVENT_EMITTER_HPP
#define LEGO_EVENT_EMITTER_HPP
#include <functional>
#include <memory>
#include <unordered_set>
namespace tg {

  /**
   * \addtogroup utilities
   * @{
   */

  /**
   * \brief Represents something that can fire events.
   *
   * Event listeners can be registered, which will be triggered when an event happens.
   *
   * When you fire an event, you can optionally give this event a payload data that will be received by all triggered event listeners.
   * For example, for a mouseclick event, the position of the cursor might be an important piece of information that you want to pass to event listeners.
   *
   * This class is currently used by tg::optimizer_base for user to respond to events such as epoch completion, new best record, etc.
   *
   * \tparam payload_T The payload type associated with an event that will be passed to all event listeners.
   */
  template<typename ...payload_T>
  class event_emitter {
  public:
    using listener_t = std::function<void(payload_T...)>;
    using listener_handle_t = std::shared_ptr<listener_t>;
  private:
    std::unordered_set<listener_handle_t> listeners;
    std::unordered_set<listener_handle_t> once_listeners;
    const listener_handle_t& add_listener(const listener_handle_t& listener) {
      listeners.insert(listener);
      return listener;
    }
  public:

    /**
     * \brief Register an event listener.
     * \param listener the function to register as listener
     * \return a handle that will be used if the listener need to be unregistered later
     */
    listener_handle_t add_listener(const listener_t& listener) {
      return add_listener(std::make_shared<listener_t>(listener));
    }

    /**
     * \brief Unregister an event listener.
     * \param listener the handle of the listener to unregister
     */
    void remove_listener(const listener_handle_t& listener) {
      listeners.erase(listener);
      once_listeners.erase(listener);
    }

    /**
     * \brief Trigger the event, invoking all registered listeners.
     *
     * There is no order guarantee of the invocations.
     *
     * \param payloads the event payloads that will be passed to all listener functions.
     */
    void fire(payload_T ...payloads) const {
      // call listeners
      {
        // make a copy so that removing listeners within listener is safe
        std::unordered_set<listener_handle_t> listeners_cp(listeners);
        for(auto&& listener:listeners_cp) {
          (*listener)(payloads...);
        }
      }
    }
  };

  /// @}
}

#endif //LEGO_EVENT_EMITTER_HPP

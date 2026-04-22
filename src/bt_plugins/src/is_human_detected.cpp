
#include "behaviortree_cpp_v3/condition_node.h"
#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include "behaviortree_cpp_v3/bt_factory.h"

namespace bt_plugins
{

class IsHumanDetected : public BT::ConditionNode
{
public:
  IsHumanDetected(const std::string & name, const BT::NodeConfiguration & config)
  : BT::ConditionNode(name, config),
    human_detected_(false),
    topic_name_("/is_human_detected")
  {
    // Node handle will be initialized in tick() via blackboard
  }

  static BT::PortsList providedPorts()
  {
    return {
      BT::InputPort<std::string>("topic_name", "/is_human_detected", "Topic to check for human detection")
    };
  }

  BT::NodeStatus tick() override
  {
    // One-time initialization when ticked for the first time
    if (!node_) {
      initialize();
    }

    // Process any incoming messages
    callback_group_executor_.spin_some();

    RCLCPP_INFO(node_->get_logger(), "IsHumanDetected: %s", human_detected_ ? "YES" : "NO");
    return human_detected_ ? BT::NodeStatus::SUCCESS : BT::NodeStatus::FAILURE;
  }

private:
  void initialize()
  {
    // Get the topic name from the input port (if specified)
    getInput("topic_name", topic_name_);

    // Get shared node from blackboard
    node_ = config().blackboard->get<rclcpp::Node::SharedPtr>("node");

    // Create callback group and executor for non-blocking spin
    callback_group_ = node_->create_callback_group(
      rclcpp::CallbackGroupType::MutuallyExclusive);
    callback_group_executor_.add_callback_group(callback_group_, node_->get_node_base_interface());

    // Setup subscription
    rclcpp::SubscriptionOptions options;
    options.callback_group = callback_group_;

    subscription_ = node_->create_subscription<std_msgs::msg::Bool>(
      topic_name_, 10,
      std::bind(&IsHumanDetected::humanCallback, this, std::placeholders::_1),
      options);

    RCLCPP_INFO(node_->get_logger(), "Subscribed to topic: %s", topic_name_.c_str());
  }

  void humanCallback(const std_msgs::msg::Bool::SharedPtr msg)
  {
    human_detected_ = msg->data;
  }

private:
  rclcpp::Node::SharedPtr node_;
  rclcpp::CallbackGroup::SharedPtr callback_group_;
  rclcpp::executors::SingleThreadedExecutor callback_group_executor_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr subscription_;

  std::atomic_bool human_detected_;
  std::string topic_name_;
};

}  

// namespace bt_plugins

// Register the plugin


BT_REGISTER_NODES(factory)
{
  BT::NodeBuilder builder = [](const std::string & name, const BT::NodeConfiguration & config) {
    return std::make_unique<bt_plugins::IsHumanDetected>(name, config);
  };

  factory.registerBuilder<bt_plugins::IsHumanDetected>("IsHumanDetected", builder);
}



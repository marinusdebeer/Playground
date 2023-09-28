class Dog
  def initialize(name)
    @name = name
  end

  def bark
    puts "#{@name} says Woof!"
  end
end

dog = Dog.new("Rex")
dog.bark  # Output: Rex says Woof!

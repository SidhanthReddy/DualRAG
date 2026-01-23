"""
State-RAG Evaluation Dataset
Sample tasks for conference paper evaluation
"""

SAMPLE_TASKS = [
    # =========================================================================
    # SIMPLE TASKS (10 tasks)
    # =========================================================================
    {
        "task_id": "simple_001",
        "name": "Button Component",
        "description": "Create a reusable React button component that accepts onClick handler, children, and optional className prop. Use Tailwind CSS for styling.",
        "complexity": "simple",
        "requirements": [
            "Create components/Button.tsx",
            "Props: onClick (function), children (ReactNode), className (optional string)",
            "Default styling: bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded",
            "Export as default"
        ],
        "ground_truth": """
import React from 'react';

interface ButtonProps {
  onClick: () => void;
  children: React.ReactNode;
  className?: string;
}

export default function Button({ onClick, children, className = '' }: ButtonProps) {
  return (
    <button
      onClick={onClick}
      className={`bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded ${className}`}
    >
      {children}
    </button>
  );
}
""",
        "test_cases": [
            "Renders children correctly",
            "Calls onClick when clicked",
            "Applies custom className"
        ]
    },
    
    {
        "task_id": "simple_002",
        "name": "Card Component",
        "description": "Create a Card component with optional title and footer sections.",
        "complexity": "simple",
        "requirements": [
            "Create components/Card.tsx",
            "Props: title (optional string), children (ReactNode), footer (optional ReactNode)",
            "Use Tailwind: border, rounded-lg, shadow-md, padding"
        ],
        "ground_truth": """
import React from 'react';

interface CardProps {
  title?: string;
  children: React.ReactNode;
  footer?: React.ReactNode;
}

export default function Card({ title, children, footer }: CardProps) {
  return (
    <div className="border rounded-lg shadow-md p-6">
      {title && <h2 className="text-xl font-bold mb-4">{title}</h2>}
      <div className="mb-4">{children}</div>
      {footer && <div className="border-t pt-4">{footer}</div>}
    </div>
  );
}
""",
        "test_cases": [
            "Renders title when provided",
            "Renders children",
            "Renders footer when provided"
        ]
    },
    
    {
        "task_id": "simple_003",
        "name": "Input Component",
        "description": "Create a controlled input component with label and error message support.",
        "complexity": "simple",
        "requirements": [
            "Create components/Input.tsx",
            "Props: label, value, onChange, error (optional), placeholder (optional)",
            "Show error message in red text below input"
        ],
        "ground_truth": """
import React from 'react';

interface InputProps {
  label: string;
  value: string;
  onChange: (value: string) => void;
  error?: string;
  placeholder?: string;
}

export default function Input({ label, value, onChange, error, placeholder }: InputProps) {
  return (
    <div className="mb-4">
      <label className="block text-sm font-medium mb-2">{label}</label>
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className={`w-full px-3 py-2 border rounded ${error ? 'border-red-500' : 'border-gray-300'}`}
      />
      {error && <p className="text-red-500 text-sm mt-1">{error}</p>}
    </div>
  );
}
""",
        "test_cases": [
            "Displays label",
            "Calls onChange with new value",
            "Shows error message when provided"
        ]
    },
    
    # =========================================================================
    # MEDIUM TASKS (20 tasks)
    # =========================================================================
    {
        "task_id": "medium_001",
        "name": "Todo App",
        "description": "Create a complete todo app with TodoList, TodoItem, and AddTodo components. Include add, toggle, and delete functionality.",
        "complexity": "medium",
        "requirements": [
            "Create components/TodoList.tsx - displays all todos",
            "Create components/TodoItem.tsx - individual todo with checkbox and delete",
            "Create components/AddTodo.tsx - form to add new todos",
            "Use useState for state management",
            "Todo interface: { id: string, text: string, completed: boolean }"
        ],
        "ground_truth": {
            "components/TodoItem.tsx": """
import React from 'react';

interface Todo {
  id: string;
  text: string;
  completed: boolean;
}

interface TodoItemProps {
  todo: Todo;
  onToggle: (id: string) => void;
  onDelete: (id: string) => void;
}

export default function TodoItem({ todo, onToggle, onDelete }: TodoItemProps) {
  return (
    <div className="flex items-center gap-3 p-3 border-b">
      <input
        type="checkbox"
        checked={todo.completed}
        onChange={() => onToggle(todo.id)}
        className="w-5 h-5"
      />
      <span className={todo.completed ? 'line-through text-gray-500' : ''}>
        {todo.text}
      </span>
      <button
        onClick={() => onDelete(todo.id)}
        className="ml-auto text-red-500 hover:text-red-700"
      >
        Delete
      </button>
    </div>
  );
}
""",
            "components/AddTodo.tsx": """
import React, { useState } from 'react';

interface AddTodoProps {
  onAdd: (text: string) => void;
}

export default function AddTodo({ onAdd }: AddTodoProps) {
  const [text, setText] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (text.trim()) {
      onAdd(text);
      setText('');
    }
  };

  return (
    <form onSubmit={handleSubmit} className="flex gap-2 mb-4">
      <input
        type="text"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Add a new todo..."
        className="flex-1 px-3 py-2 border rounded"
      />
      <button type="submit" className="bg-blue-500 text-white px-4 py-2 rounded">
        Add
      </button>
    </form>
  );
}
""",
            "components/TodoList.tsx": """
import React, { useState } from 'react';
import TodoItem from './TodoItem';
import AddTodo from './AddTodo';

interface Todo {
  id: string;
  text: string;
  completed: boolean;
}

export default function TodoList() {
  const [todos, setTodos] = useState<Todo[]>([]);

  const addTodo = (text: string) => {
    const newTodo: Todo = {
      id: Date.now().toString(),
      text,
      completed: false
    };
    setTodos([...todos, newTodo]);
  };

  const toggleTodo = (id: string) => {
    setTodos(todos.map(todo =>
      todo.id === id ? { ...todo, completed: !todo.completed } : todo
    ));
  };

  const deleteTodo = (id: string) => {
    setTodos(todos.filter(todo => todo.id !== id));
  };

  return (
    <div className="max-w-md mx-auto mt-8 p-6 bg-white rounded-lg shadow">
      <h1 className="text-2xl font-bold mb-4">Todo List</h1>
      <AddTodo onAdd={addTodo} />
      <div>
        {todos.map(todo => (
          <TodoItem
            key={todo.id}
            todo={todo}
            onToggle={toggleTodo}
            onDelete={deleteTodo}
          />
        ))}
      </div>
    </div>
  );
}
"""
        },
        "test_cases": [
            "Can add new todos",
            "Can toggle todo completion",
            "Can delete todos",
            "Todos persist across re-renders"
        ]
    },
    
    {
        "task_id": "medium_002",
        "name": "User Profile Form",
        "description": "Create a user profile form with validation for name, email, and bio fields.",
        "complexity": "medium",
        "requirements": [
            "Create components/ProfileForm.tsx",
            "Fields: name (required), email (required, valid format), bio (optional)",
            "Show validation errors",
            "Submit button disabled until valid",
            "onSubmit callback prop"
        ],
        "dependencies": ["components/Input.tsx"],
        "user_edits": [
            {
                "file": "components/Input.tsx",
                "edit": "Add ARIA label for accessibility",
                "code": "aria-label={label}"
            }
        ],
        "test_cases": [
            "Validates email format",
            "Shows error messages",
            "Calls onSubmit with form data",
            "Preserves ARIA labels from Input component"
        ]
    },
    
    # =========================================================================
    # COMPLEX TASKS (20 tasks)
    # =========================================================================
    {
        "task_id": "complex_001",
        "name": "E-commerce Product Listing",
        "description": "Create a product listing page with search, filtering, sorting, and cart functionality.",
        "complexity": "complex",
        "requirements": [
            "Create components/ProductCard.tsx - individual product display",
            "Create components/ProductList.tsx - grid of products",
            "Create components/SearchBar.tsx - search input",
            "Create components/FilterPanel.tsx - category and price filters",
            "Create components/Cart.tsx - shopping cart sidebar",
            "Create hooks/useCart.ts - cart state management",
            "Create types/Product.ts - product interface",
            "Features: search, filter by category, sort by price, add to cart"
        ],
        "ground_truth": {
            # Multiple files here...
        },
        "user_edits": [
            {
                "step": 3,
                "file": "components/ProductCard.tsx",
                "edit": "User adds analytics tracking",
                "code": "onClick={() => { analytics.track('product_clicked', product.id); onAddToCart(product); }}"
            },
            {
                "step": 7,
                "file": "hooks/useCart.ts",
                "edit": "User optimizes cart performance",
                "code": "Use useMemo for cart total calculation"
            }
        ],
        "test_cases": [
            "Search filters products correctly",
            "Filters work together (AND logic)",
            "Cart adds/removes items",
            "Cart total calculates correctly",
            "User analytics code preserved after AI modifications",
            "User performance optimization preserved"
        ]
    },
    
    # =========================================================================
    # AUTHORITY PRESERVATION TEST CASES
    # =========================================================================
    {
        "task_id": "authority_001",
        "name": "Security Patch Preservation",
        "description": "Test if AI preserves critical security fixes during refactoring",
        "complexity": "medium",
        "scenario": [
            {
                "step": 1,
                "action": "AI creates Button component",
                "files": ["components/Button.tsx"]
            },
            {
                "step": 2,
                "action": "User adds security sanitization",
                "file": "components/Button.tsx",
                "code": "const sanitize = (input: string) => DOMPurify.sanitize(input);"
            },
            {
                "step": 3,
                "action": "AI request: Add loading state to button",
                "expected": "Security sanitization code must be preserved"
            }
        ],
        "success_criteria": "User's sanitization code present in final output"
    },
    
    # =========================================================================
    # DEPENDENCY TRACKING TEST CASES
    # =========================================================================
    {
        "task_id": "dependency_001",
        "name": "Transitive Dependency Inclusion",
        "description": "Test if AI includes all necessary dependencies",
        "complexity": "medium",
        "dependency_graph": {
            "Button": ["Icon", "Theme"],
            "Card": ["Button", "Image"],
            "Form": ["Card", "Input", "Button"],
            "Page": ["Form", "Navbar", "Footer"]
        },
        "test": {
            "request": "Modify the Page component",
            "expected_includes": ["Page", "Form", "Card", "Button", "Input", "Navbar", "Footer", "Icon", "Theme", "Image"],
            "metric": "Recall: % of true dependencies included"
        }
    },
    
    # =========================================================================
    # COST COMPARISON TEST CASES
    # =========================================================================
    {
        "task_id": "cost_001",
        "name": "10-Step Workflow Token Usage",
        "description": "Measure token usage across iterative development",
        "complexity": "complex",
        "workflow": [
            "Create Navbar",
            "Create Home page",
            "Create About page",
            "Update Navbar with links",
            "User fixes Navbar spacing",
            "Create Contact page",
            "Update Navbar with Contact link",
            "Create Footer",
            "User adds analytics to Footer",
            "Refactor Navbar for mobile"
        ],
        "metrics": [
            "Total tokens used",
            "Tokens per step",
            "Final cost in USD"
        ]
    }
]


# =============================================================================
# Helper Functions
# =============================================================================

def get_tasks_by_complexity(complexity: str):
    """Get all tasks of a given complexity level"""
    return [task for task in SAMPLE_TASKS if task['complexity'] == complexity]


def get_simple_tasks():
    return get_tasks_by_complexity('simple')


def get_medium_tasks():
    return get_tasks_by_complexity('medium')


def get_complex_tasks():
    return get_tasks_by_complexity('complex')


def get_authority_tests():
    return [task for task in SAMPLE_TASKS if 'authority' in task['task_id']]


def get_dependency_tests():
    return [task for task in SAMPLE_TASKS if 'dependency' in task['task_id']]


def get_cost_tests():
    return [task for task in SAMPLE_TASKS if 'cost' in task['task_id']]


if __name__ == "__main__":
    print(f"Total tasks: {len(SAMPLE_TASKS)}")
    print(f"Simple: {len(get_simple_tasks())}")
    print(f"Medium: {len(get_medium_tasks())}")
    print(f"Complex: {len(get_complex_tasks())}")
    print(f"Authority tests: {len(get_authority_tests())}")
    print(f"Dependency tests: {len(get_dependency_tests())}")
    print(f"Cost tests: {len(get_cost_tests())}")
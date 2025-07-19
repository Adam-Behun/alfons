import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

export function formatTimestamp(timestamp: string): string {
  return new Date(timestamp).toLocaleTimeString()
}

export function formatDate(dateString: string): string {
  return new Date(dateString).toLocaleDateString()
}

export function getStatusColor(status: string): string {
  switch (status.toLowerCase()) {
    case 'success':
    case 'approved':
      return 'text-green-600 bg-green-50'
    case 'failure':
    case 'denied':
      return 'text-red-600 bg-red-50'
    case 'pending':
      return 'text-amber-600 bg-amber-50'
    default:
      return 'text-gray-600 bg-gray-50'
  }
}
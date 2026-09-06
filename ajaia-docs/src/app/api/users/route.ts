import { NextResponse } from 'next/server';
import { getAllUsers } from '@/lib/users';

export async function GET() {
  return NextResponse.json({ success: true, users: getAllUsers() });
}
